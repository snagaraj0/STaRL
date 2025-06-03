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
import random
import argparse

"""
TEMP = 0.7
MAX_NEW_TOKENS = 512
BATCH_SIZE = 4
NUM_EPOCHS = 5
TRAIN_SIZE = 5000
VAL_SIZE = 500
LOG_EVERY = 10
LOG_EVERY = 50
BATCH_SIZE = 5
LOG_JSON_DIR = "batch_logs"
"""
device = "cuda" if torch.cuda.is_available() else "cpu"

def parse_args():
    parser = argparse.ArgumentParser(description="Training Configuration")

    parser.add_argument("--temp", type=float, default=0.7, help="Sampling temperature")
    parser.add_argument("--max_new_tokens", type=int, default=512, help="Max new tokens to generate")
    parser.add_argument("--batch_size", type=int, default=5, help="Batch size")
    parser.add_argument("--num_epochs", type=int, default=5, help="Number of training epochs")
    parser.add_argument("--train_size", type=int, default=5000, help="Training set size")
    parser.add_argument("--val_size", type=int, default=150, help="Validation set size")
    parser.add_argument("--log_every", type=int, default=50, help="Steps between logs")
    parser.add_argument("--log_json_dir", type=str, default="test_exp/batch_logs", help="Directory to save log JSONs")
    parser.add_argument("--save_checkpoint_every", type=int, default=100, help="How frequently to save model checkpoints")

    return parser.parse_args()

def setup(log_json_dir):
    os.makedirs(log_json_dir, exist_ok=True)
    os.makedirs(os.path.join(log_json_dir, "train"), exist_ok=True)
    os.makedirs(os.path.join(log_json_dir, "val"), exist_ok=True)
    os.makedirs(os.path.join(log_json_dir, "checkpoints"), exist_ok=True)

def get_model(model_name):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name).to(device)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        model.config.pad_token_id = model.config.eos_token_id
    tokenizer.padding_side = "left"
    return model, tokenizer

class CommonsenseQAParser:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        self.system_prompt = """You are an expert at applying commonsense reasoning to answer multiple-choice questions. 
                        For each question:
                        1. Identify what the question is asking for
                        2. Evaluate each choice against the question's requirements
                        3. Select the choice that best fits

                        Examples:

                        Q: What do people use to absorb extra ink from a fountain pen?
                        Answer Choices: (a) shirt pocket (b) calligrapher's hand (c) inkwell (d) desk drawer (e) blotter
                        A: The question asks for something that absorbs ink. A blotter is specifically designed to absorb excess ink from writing instruments. Therefore, the answer is blotter (e).

                        Q: What home entertainment equipment requires cable?
                        Answer Choices: (a) radio shack (b) substation (c) television (d) cabinet (e) desk
                        A: The question asks for entertainment equipment that needs cable service. Televisions can receive cable TV signals for entertainment programming. Therefore, the answer is television (c).

                        Always format your response as:
                        "<Clear reasoning in 1-2 sentences>. Therefore, the answer is <answer text> (<letter>)."

                        Choose the most reasonable answer if uncertain."""
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
def sample_no_grad(model, tokenizer, batch_prompt_ids, max_new_tokens=512, temp=0.7):
    # `batch_prompt_ids` is shape (B, T)
    seq = model.generate(
        batch_prompt_ids,
        max_new_tokens=max_new_tokens,
        temperature=temp,
        do_sample=True,
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

# Base reward function
def compute_binary_reward(final_answer, correct_answer, question=None, rationale=None):
    return 1.0 if final_answer == correct_answer else -1.0


# Build batches
def get_batches(dataset, parser, batch_size):
    for i in range(0, len(dataset), batch_size):
        batch = [{k: dataset[k][j] for k in dataset} for j in range(i, min(i + batch_size, len(dataset)))]
        prompts, raw_qs, answers = zip(*[
            (*parser.format_prompt(item), item["answerKey"].lower())
            for item in batch
        ])
        yield list(prompts), list(raw_qs), list(answers)

def train(model_name, temp, max_new_tokens, batch_size, num_epochs, train_size, val_size, log_every, log_json_dir, save_checkpoint_every=100):
    model, tokenizer = get_model(model_name)
    # Subsample training to 5000 examples
    train_dataset = load_dataset("commonsense_qa", split=f"train[:{train_size}]")
    # Subsample validation (i.e. test) to 150 examples
    val_dataset = load_dataset("commonsense_qa", split=f"validation[:{val_size}]")

    writer = SummaryWriter()
    parser = CommonsenseQAParser(tokenizer)
    optimizer = optim.AdamW(model.parameters(), lr=5e-7)

    # Pre-process datasets to avoid repeated parsing
    print("Pre-processing datasets...")
    train_batches = list(get_batches(train_dataset, parser, batch_size))
    val_batches = list(get_batches(val_dataset, parser, batch_size))
    print(f"Created {len(train_batches)} training batches, {len(val_batches)} validation batches")

    # Reduce validation frequency significantly
    VALIDATION_FREQUENCY = len(train_batches) // 5

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
                compute_binary_reward(ans, corr)
                for ans, corr in zip(pred_answers, correct_answers)
            ], device=device, dtype=torch.float32)
            
            # REINFORCE loss
            loss = -(rewards * logprobs).mean()

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
                ckpt_path = os.path.join(log_json_dir, "checkpoints", f"epoch{epoch + 1}_batch{batch_idx}")
                os.makedirs(os.path.dirname(ckpt_path), exist_ok=True)
                model.save_pretrained(ckpt_path)
                tokenizer.save_pretrained(ckpt_path)

            ### VALIDATION LOGGING (Much less frequent) ###
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
                            max_length=512
                        ).input_ids.to(device)

                        val_gen_ids = sample_no_grad(model, val_prompt_ids, 
                                                max_new_tokens=max_new_tokens, temp=temp)
                        val_gen_strs = tokenizer.batch_decode(val_gen_ids, skip_special_tokens=True)
                        
                        val_parsed = [parser.parse_llm_output(gen) for gen in val_gen_strs]
                        val_rationales, val_pred_answers = zip(*val_parsed)
                        
                        val_logp = compute_logprobs(model, val_prompt_ids, val_gen_ids, temp=temp)
                        val_rwds = torch.tensor([
                            compute_binary_reward(ans, corr)
                            for ans, corr in zip(val_pred_answers, val_correct_answers)
                        ], device=device, dtype=torch.float32)

                        val_loss = -(val_rwds * val_logp).mean()

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
                
                model.train()  # Return to training mode
                torch.cuda.empty_cache()  # Clean up after validation

        avg_loss = train_loss / len(train_batches)
        print(f"Epoch {epoch + 1} avg train loss per batch: {avg_loss:.4f}")
        writer.add_scalar("Loss/train", avg_loss, epoch + 1)

    writer.close()

    

if __name__ == "__main__":
    model_name = "Qwen/Qwen2.5-0.5B-Instruct" #Qwen/Qwen3-1.7B
    args = parse_args()
    setup(args.log_json_dir)
    train(model_name, args.temp, args.max_new_tokens, args.batch_size, args.num_epochs, args.train_size,
          args.val_size, args.log_every, args.log_json_dir, args.save_checkpoint_every)
