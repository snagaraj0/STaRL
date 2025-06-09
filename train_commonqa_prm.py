import torch
import torch.nn.functional as F
from torch import optim
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
import json
import os
import re
import shutil
import random
import argparse
import pprint
import openai


device = "cuda" if torch.cuda.is_available() else "cpu"

def parse_args():
    parser = argparse.ArgumentParser(description="Training Configuration")
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen2.5-0.5B-Instruct", help = "Which HF model to use")
    parser.add_argument("--reward_type", type=str, choices=["lm", "normalized_lm", "normalized_lm_variance"], default="lm", 
                        help="Type of reward to use. Choose from: lm, normalized_lm, normalized_lm_variance")
    parser.add_argument("--temp", type=float, default=0.7, help="Sampling temperature")
    parser.add_argument("--max_new_tokens", type=int, default=200, help="Max new tokens to generate")
    parser.add_argument("--batch_size", type=int, default=5, help="Batch size")
    parser.add_argument("--num_epochs", type=int, default=5, help="Number of training epochs")
    parser.add_argument("--train_size", type=int, default=5000, help="Training set size")
    parser.add_argument("--val_size", type=int, default=150, help="Validation set size")
    parser.add_argument("--log_every", type=int, default=50, help="Steps between logs")
    parser.add_argument("--out_dir", type=str, default="test_exp", help="Directory to save run outputs to")
    parser.add_argument("--save_checkpoint_every", type=int, default=500, help="How frequently to save model checkpoints")
    parser.add_argument("--lr", type=float, default=5e-7, help="LR for model")

    return parser.parse_args()

def setup(log_json_dir):
    os.makedirs(log_json_dir, exist_ok=True)
    os.makedirs(os.path.join(log_json_dir, "train"), exist_ok=True)
    os.makedirs(os.path.join(log_json_dir, "val"), exist_ok=True)

def get_model(model_name):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name).to(device)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = "<|reserved_special_token_4|>"
    tokenizer.padding_side = "left"
    return model, tokenizer

class CommonsenseQAParser:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        self.system_prompt = """"""

    def format_question(self, question_data):
        q = question_data['question']
        choices = "".join(
            f"({lbl.lower()}) {txt}\n"
            for lbl, txt in zip(
                question_data['choices']['label'], question_data['choices']['text']
            )
        )
        return f"Q: {q}\nAnswer Choices:\n{choices.strip()}\nA: "

    def format_prompt(self, question_data):
        messages = [
            {"role": "system",  "content": self.system_prompt},
            {"role": "user",    "content": self.format_question(question_data)}
        ]
        return self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False,
            enable_thinking=False
        ), messages[-1]['content']

    def parse_llm_output(self, generated_text):
        rationale = generated_text.removeprefix("</think>").strip()
        matches = re.findall(r"\(([a-e])\)", generated_text, re.IGNORECASE)
        letter = matches[-1].lower() if matches else None
        return rationale, letter


@torch.no_grad()
def sample_no_grad(model, tokenizer, batch_prompt_ids, max_new_tokens=256, temp=0.7):
    # `batch_prompt_ids` is shape (B, T)
    seq = model.generate(
        batch_prompt_ids,
        max_new_tokens=max_new_tokens,
        temperature=temp,
        do_sample=True,
        repetition_penalty=1.2,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id
    )
    # We return only the newly generated portion
    return seq[:, batch_prompt_ids.size(1):]  # shape (B, new_T)


def compute_logprobs(model, tokenizer, batch_prompt_ids, batch_gen_ids, temp=0.7):
    # Concatenate inputs
    full_ids = torch.cat([batch_prompt_ids, batch_gen_ids], dim=1)  # (B, T+new_T)

    # Forward pass
    full_logits = model(full_ids).logits / temp  # (B, T+new_T, V)
    full_logprobs = F.log_softmax(full_logits, dim=-1)  # (B, T+new_T, V)

    # Get logprobs of actual tokens (predict token_t at position t)
    token_logprobs = full_logprobs[:, :-1, :].gather(
        2, full_ids[:, 1:].unsqueeze(-1)
    ).squeeze(-1)  # (B, T+new_T - 1)

    # Lengths
    prompt_lens = (batch_prompt_ids != tokenizer.pad_token_id).sum(dim=1)  # (B,)
    gen_lens = (batch_gen_ids != tokenizer.pad_token_id).sum(dim=1)        # (B,)

    # Create a mask for the generated tokens
    _, total_len = full_ids.size()
    token_pos = torch.arange(total_len - 1, device=full_ids.device).unsqueeze(0)  # (1, T+new_T - 1)
    gen_masks = (
        (token_pos >= prompt_lens.unsqueeze(1)) &
        (token_pos < (prompt_lens + gen_lens).unsqueeze(1))
    )  # (B, T+new_T - 1), True where generated tokens live

    # Mask and sum
    gen_token_logprobs = token_logprobs * gen_masks  # (B, T+new_T - 1)
    return gen_token_logprobs.sum(dim=1)  # (B,)

# PRM reward function
def compute_LM_reward(correct_answer, question, rationale):
    api_key = "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"
    client = openai.OpenAI(api_key=api_key)
    
    prompt = f"""Given the below question, whose correct answer is ({correct_answer}), assign a numerical score to between -1 (poor) and 1 (excellent) to the given response. Score the response based on the quality of reasoning, logical coherence and consistency, and accuracy in answering the question.

    Provide ONLY a decimal score between -1 (poor) and 1 (excellent), rounded to the nearest hundredth. Your answer should be "X.XX", without quotes or any explanation.

    QUESTION: {question}
    RESPONSE: {rationale}
    """
    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini", messages=[{"role": "user", "content": prompt}], max_tokens=10
        )

        score_text = response.choices[0].message.content.strip()
        return float(score_text)
    except Exception as e:
        print(f"Error during API call: {e}")
        # If prompting fails, reward is equivalent to an incorrect answer
        return -1.0


# Build batches
def get_batches(dataset, parser, batch_size):
    # Returns (prompt_strs, raw_qs, correct_answers), each a len-B List[str].
    for i in range(0, len(dataset), batch_size):
        batch_dict = dataset[i : i + batch_size]
        batch_items = [
            {key: batch_dict[key][i] for key in batch_dict}
            for i in range(len(batch_dict["id"]))
        ]
        prompt_strs, raw_qs, correct_keys = [], [], []
        for item in batch_items:
            p_str, raw_q = parser.format_prompt(item)
            prompt_strs.append(p_str)
            raw_qs.append(raw_q)
            correct_keys.append(item["answerKey"].lower())
        yield prompt_strs, raw_qs, correct_keys

def train(model_name, temp, max_new_tokens, batch_size, num_epochs, train_size, val_size, log_every, out_dir, 
          save_checkpoint_every=100, reward_type="binary", lr=5e-7):
    log_json_dir = os.path.join(args.out_dir, "batch_logs")
    model, tokenizer = get_model(model_name)
    # Subsample training to 5000 examples
    train_dataset = load_dataset("commonsense_qa", split=f"train[:{train_size}]")
    # Subsample validation/test to 150 examples
    val_dataset = load_dataset("commonsense_qa", split=f"validation[:{val_size}]")

    writer = SummaryWriter(out_dir)
    parser = CommonsenseQAParser(tokenizer)
    optimizer = optim.AdamW(model.parameters(), lr=lr)

    # Pre-process datasets to avoid repeated parsing
    print("Pre-processing datasets...")
    train_batches = list(get_batches(train_dataset, parser, batch_size))
    val_batches = list(get_batches(val_dataset, parser, batch_size))
    print(f"Created {len(train_batches)} training batches, {len(val_batches)} validation batches")

    # Reduce validation frequency significantly
    VALIDATION_FREQUENCY = len(train_batches) // 2  

    for epoch in range(num_epochs):
        ##### TRAINING LOOP #####
        model.train()
        train_loss = 0.0
        
        for batch_idx, (prompt_strs, raw_qs, correct_answers) in enumerate(tqdm(
            train_batches, desc=f"Training Epoch {epoch + 1}"
        )):
            # Tokenize once with proper device placement
            prompt_ids = tokenizer(
                prompt_strs, 
                return_tensors="pt",
                padding=True, 
                truncation=True, 
                max_length=512
            ).input_ids.to(device)

            # Generate samples
            gen_ids = sample_no_grad(
                model, tokenizer, prompt_ids,
                max_new_tokens=max_new_tokens, temp=temp
            )

            # Batch decode and parse
            gen_strs = tokenizer.batch_decode(gen_ids, skip_special_tokens=True)
            parsed_outputs = [parser.parse_llm_output(gen_str) for gen_str in gen_strs]
            rationales, pred_answers = zip(*parsed_outputs)

            # Compute loss components
            logprobs = compute_logprobs(model, tokenizer, prompt_ids, gen_ids, temp=temp)
            rewards = torch.tensor([
                compute_LM_reward(ans, question, rationale)
                for ans, question, rationale in zip(correct_answers, raw_qs, rationales)
            ], device=device, dtype=torch.float32)
            
            # REINFORCE loss
            if reward_type == "lm":
                loss = -(rewards * logprobs).mean()
            # REINFORCE (baseline-normalized) loss
            elif reward_type == "normalized_lm":
                baseline = rewards.mean()
                loss = -((rewards - baseline) * logprobs).mean()
            elif reward_type == "normalized_lm_variance":
                baseline = rewards.mean()
                variance = rewards.var(unbiased=False) + 1e-8 
                loss = -((rewards - baseline) / variance * logprobs).mean()
            else:
                raise ValueError(f"Invalid reward_type: {reward_type}.")
            
            # Gradient step
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()

            # Lightweight logging (reduced frequency)
            if batch_idx % log_every == 0:
                writer.add_scalar("BatchLoss/train", loss.item(), epoch * len(train_batches) + batch_idx)
                writer.add_scalar("BatchAvgLogProb/train", logprobs.mean().item(), epoch * len(train_batches) + batch_idx)
                writer.add_scalar("BatchAvgReward/train", rewards.mean().item(), epoch * len(train_batches) + batch_idx)
                
                # Simplified logging - only save one example per LOG_EVERY batches
                log_data = {
                    "epoch": epoch + 1,
                    "batch": batch_idx,
                    "question": raw_qs[0],
                    "full_llm_output": gen_strs[0],
                    "rationale": rationales[0],
                    "predicted_answer": pred_answers[0] if pred_answers[0] else "None",
                    "correct_answer": correct_answers[0],
                    "reward": float(rewards[0].item()),
                    "logprob": float(logprobs[0].item()),
                }
                json_path = os.path.join(log_json_dir, "train", f"epoch{epoch + 1}_batch{batch_idx}.json")
                os.makedirs(os.path.dirname(json_path), exist_ok=True)
                with open(json_path, "w") as f:
                    json.dump(log_data, f, indent=2)

            # Aggressive memory cleanup
            del prompt_ids, gen_ids, logprobs, rewards, loss
            if batch_idx % 10 == 0:  # Clean cache every 10 batches
                torch.cuda.empty_cache()
            
            if batch_idx % save_checkpoint_every == 0:
                ckpt_path = os.path.join(out_dir, "checkpoints", f"epoch{epoch + 1}_batch{batch_idx}")
                os.makedirs(os.path.dirname(ckpt_path), exist_ok=True)
                model.save_pretrained(ckpt_path)
                tokenizer.save_pretrained(ckpt_path)

            ### VALIDATION LOGGING ###
            if batch_idx > 0 and batch_idx % VALIDATION_FREQUENCY == 0:
                print(f"\n--- Running validation at batch {batch_idx} ---")
                model.eval()
                
                val_rewards, val_logprobs, val_losses = [], [], []
                collected_examples = []

                with torch.no_grad():
                    for val_batch in val_batches:
                        val_prompt_strs, val_raw_qs, val_correct_answers = val_batch
                        
                        val_prompt_ids = tokenizer(
                            val_prompt_strs, 
                            return_tensors="pt",
                            padding=True, 
                            truncation=True, 
                            max_length=max_new_tokens
                        ).input_ids.to(device)

                        val_gen_ids = sample_no_grad(model, tokenizer, val_prompt_ids, 
                                                max_new_tokens=max_new_tokens, temp=temp)
                        val_gen_strs = tokenizer.batch_decode(val_gen_ids, skip_special_tokens=True)
                        
                        val_parsed = [parser.parse_llm_output(gen) for gen in val_gen_strs]
                        val_rationales, val_pred_answers = zip(*val_parsed)
                        
                        val_logp = compute_logprobs(model, tokenizer, val_prompt_ids, val_gen_ids, temp=temp)
                        val_rwds = torch.tensor([
                             compute_LM_reward(ans, question, rationale)
                            for ans, question, rationale in zip(val_correct_answers, val_raw_qs, val_rationales)
                        ], device=device, dtype=torch.float32)

                        #val_loss = -(val_rwds * val_logp).mean()
                        val_loss = -((val_rwds - val_rwds.mean()) * val_logp).mean()

                        # REINFORCE loss
                        if reward_type == "lm":
                            val_loss = -(val_rwds * val_logp).mean()
                        elif reward_type == "normalized_lm":
                            baseline = val_rwds.mean()
                            # REINFORCE (baseline-normalized) loss
                            val_loss = -((val_rwds - baseline) * val_logp).mean()
                        elif reward_type == "normalized_lm_variance":
                            baseline = val_rwds.mean()
                            variance = val_rwds.var(unbiased=False) + 1e-8 
                            val_loss = -((val_rwds - baseline) / variance * val_logp).mean()
                        else:
                            raise ValueError(f"Invalid reward_type: {reward_type}.")


                        # Collect metrics
                        val_rewards.extend(val_rwds.cpu().tolist())
                        val_logprobs.extend(val_logp.cpu().tolist())
                        val_losses.append(val_loss.item())

                        # Collect sample examples (limit to reduce memory)
                        for i, (q, r, a, p, rew, lp) in enumerate(zip(
                            val_raw_qs[:2], val_rationales[:2], val_correct_answers[:2], 
                            val_pred_answers[:2], val_rwds[:2], val_logp[:2]
                        )):
                            collected_examples.append({
                                "question": q,
                                "rationale": r,
                                "predicted_answer": p if p else "None",
                                "correct_answer": a,
                                "reward": float(rew.item()),
                                "logprob": float(lp.item())
                            })

                        # Immediate cleanup
                        del val_prompt_ids, val_gen_ids, val_logp, val_rwds, val_loss

                # Log validation metrics
                avg_val_reward = sum(val_rewards) / len(val_rewards)
                avg_val_logprob = sum(val_logprobs) / len(val_logprobs)
                avg_val_loss = sum(val_losses) / len(val_losses)
                
                global_step = epoch * len(train_batches) + batch_idx
                writer.add_scalar("Eval/AvgReward", avg_val_reward, global_step)
                writer.add_scalar("Eval/AvgLogProb", avg_val_logprob, global_step)
                writer.add_scalar("Eval/AvgLoss", avg_val_loss, global_step)

                # Save validation examples (reduced sample size)
                log_sample = random.sample(collected_examples, k=min(3, len(collected_examples)))
                log_json = {
                    "epoch": epoch + 1,
                    "batch": batch_idx,
                    "avg_reward": avg_val_reward,
                    "avg_logprob": avg_val_logprob,
                    "avg_loss": avg_val_loss,
                    "sample_questions": [ex["question"] for ex in log_sample],
                    "sample_rationales": [ex["rationale"] for ex in log_sample],
                    "sample_predicted_answers": [ex["predicted_answer"] for ex in log_sample],
                    "sample_correct_answers": [ex["correct_answer"] for ex in log_sample],
                    "sample_rewards": [ex["reward"] for ex in log_sample],
                    "sample_logprobs": [ex["logprob"] for ex in log_sample],
                }
                eval_json_path = os.path.join(log_json_dir, "val", f"epoch_{epoch + 1}_batch_{batch_idx}.json")
                os.makedirs(os.path.dirname(eval_json_path), exist_ok=True)
                with open(eval_json_path, "w") as f:
                    json.dump(log_json, f, indent=2)
                
                model.train()
                torch.cuda.empty_cache() 

        avg_loss = train_loss / len(train_batches)
        print(f"Epoch {epoch + 1} avg train loss per batch: {avg_loss:.4f}")
        writer.add_scalar("Loss/train", avg_loss, epoch + 1)

    writer.close()

    

if __name__ == "__main__":
    args = parse_args()
    pprint.pprint(vars(args))
    if os.path.exists(args.out_dir) and os.path.isdir(args.out_dir):
        shutil.rmtree(args.out_dir)
    os.makedirs(args.out_dir, exist_ok=True)
    setup(os.path.join(args.out_dir, "batch_logs"))
    train(args.model_name, args.temp, args.max_new_tokens, args.batch_size, args.num_epochs, args.train_size,
          args.val_size, args.log_every, args.out_dir, args.save_checkpoint_every, args.reward_type, args.lr)
