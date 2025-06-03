# Utils for doing some logging on the results of the datasets (to be used later)
import json
from pathlib import Path
import re
from tqdm import tqdm
import torch
import torch.nn.functional as F
import csv

def get_unique_path(base_path: Path) -> Path:
    # return a unique file name
    if not base_path.exists():
        return base_path
    i = 1
    while True:
        new_path = base_path.with_name(f"{base_path.stem}_{i}{base_path.suffix}")
        if not new_path.exists():
            return new_path
        i += 1

def logging(results, dataset_name):
  # Save results to file
  output_dir = Path("outputs")
  output_dir.mkdir(parents=True, exist_ok=True)

  results_path = get_unique_path(output_dir / f"{dataset_name}_results.json")
  summary_path = get_unique_path(output_dir / f"{dataset_name}_summary.json")

  with open(results_path, "w") as f:
      json.dump(results, f, indent=2)

  # Compute and print accuracy
  correct = sum(r["reward"] for r in results if r["reward"] is not None)
  total = len(results)
  accuracy = correct / total if total > 0 else 0.0
  # Save summary to JSON

  summary = {
      "correct": correct,
      "total": total,
      "accuracy": accuracy
  }

  with open(summary_path, "w") as f:
      json.dump(summary, f, indent=2)
  print(f"\nAccuracy: {accuracy:.2%} ({correct}/{total})")

  print(f"Saved results to {results_path}")
  print(f"Saved summary to {summary_path}")


def generate_responses_generic(
    dataset,
    model,
    tokenizer,
    evaluator,
    device,
    max_tokens=200,
    reward_fn=None,
    verbose=True,
):
    """
    Generic function that setups up eval
    """
    results = []
    model.to(device)
    model.eval()
    
    print(f"The dataset has {len(dataset)} examples\n")

    for idx, item in enumerate(tqdm(dataset)):
        correct_answer = item.get('answerKey', '').lower()
        print(f"\nThis is example {idx + 1}\n")

        # Format input prompt using evaluator
        prompt = evaluator.format_prompt(item)
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1024).to(device)

        # Generate model output
        with torch.no_grad():
            output_ids = model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
                temperature=0.7,
                top_k=20,
                top_p=0.8,
            )

        # Extract generated text
        n_input_tokens = inputs.input_ids.shape[1]
        generated_ids = output_ids[0, n_input_tokens:]
        generated_text = tokenizer.decode(generated_ids, skip_special_tokens=True)

        # Compute log-probability of generated tokens
        logits = model(output_ids).logits[0, n_input_tokens:]
        log_probs_all = F.log_softmax(logits, dim=-1)
        token_log_probs = torch.gather(log_probs_all, 1, generated_ids.unsqueeze(1)).squeeze(1)
        total_log_prob = token_log_probs.sum().item()

        # Parse output
        rationale, final_answer = evaluator.parse_llm_output(generated_text)

        # Compute reward
        reward = reward_fn(final_answer, correct_answer) if reward_fn else None

        if verbose:
            print("\n--------------------")
            print(f"QUESTION:\n{evaluator.format_question(item)}")
            print(f"PROMPT:\n{prompt}")
            print(f"\nGENERATION:\n{generated_text}")
            print(f"\nPREDICTION: {final_answer}, TRUE ANSWER: {correct_answer}, REWARD: {reward}")
            print(f"logP(y | x) = {total_log_prob:.3f}")

        results.append({
            "question": evaluator.format_question(item),
            "prompt": prompt,
            "generated_text": generated_text,
            "final_answer": final_answer,
            "correct_answer": correct_answer,
            "rationale": rationale,
            "reward": reward,
            "logprob": total_log_prob,
        })

    return results


def strip_html(text):
    """Remove HTML tags like <span class='from'> and <span class='to'>."""
    return re.sub(r"<[^>]+>", "", text)

def load_turk_data(file_path):
    examples = []

    with open(file_path, newline='', encoding='utf-8') as csvfile:
        reader = csv.DictReader(csvfile)

        for row in reader:
            # Collect training input/output pairs
            train_pairs = []
            for i in range(8):
                before = strip_html(row[f'before_{i}'])
                after = strip_html(row[f'after_{i}'])
                train_pairs.append((before, after))

            # Collect test input/output
            test_before = strip_html(row['test_before'])
            test_after = strip_html(row['test_after'])

            # Collect hint if available
            hint = strip_html(row['hint']).strip()

            # Store everything
            examples.append({
                "example_pairs": train_pairs,
                "test_query": test_before,
                "answerKey": test_after,
                "hint": hint if hint else None
            })

    return examples



def generate_responses_generic(
    dataset,
    model,
    tokenizer,
    evaluator,
    device,
    max_tokens=200,
    reward_fn=None,
    verbose=True,
):
    """
    Generic function that setups up eval
    """
    results = []
    model.to(device)
    model.eval()
    
    print(f"The dataset has {len(dataset)} examples\n")

    for idx, item in enumerate(tqdm(dataset)):
        correct_answer = item.get('answerKey', '').lower()
        print(f"\nThis is example {idx + 1}\n")

        # Format input prompt using evaluator
        prompt = evaluator.format_prompt(item)
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1024).to(device)

        # Generate model output
        with torch.no_grad():
            output_ids = model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
                temperature=0.7,
                top_k=20,
                top_p=0.8,
            )

        # Extract generated text
        n_input_tokens = inputs.input_ids.shape[1]
        generated_ids = output_ids[0, n_input_tokens:]
        generated_text = tokenizer.decode(generated_ids, skip_special_tokens=True)

        # Compute log-probability of generated tokens
        logits = model(output_ids).logits[0, n_input_tokens:]
        log_probs_all = F.log_softmax(logits, dim=-1)
        token_log_probs = torch.gather(log_probs_all, 1, generated_ids.unsqueeze(1)).squeeze(1)
        total_log_prob = token_log_probs.sum().item()

        # Parse output
        rationale, final_answer = evaluator.parse_llm_output(generated_text)

        # Compute reward
        reward = reward_fn(final_answer, correct_answer) if reward_fn else None

        if verbose:
            print("\n--------------------")
            print(f"QUESTION:\n{evaluator.format_question(item)}")
            print(f"PROMPT:\n{prompt}")
            print(f"\nGENERATION:\n{generated_text}")
            print(f"\nPREDICTION: {final_answer}, TRUE ANSWER: {correct_answer}, REWARD: {reward}")
            print(f"logP(y | x) = {total_log_prob:.3f}")

        results.append({
            "question": evaluator.format_question(item),
            "prompt": prompt,
            "generated_text": generated_text,
            "final_answer": final_answer,
            "correct_answer": correct_answer,
            "rationale": rationale,
            "reward": reward,
            "logprob": total_log_prob,
        })

    return results
