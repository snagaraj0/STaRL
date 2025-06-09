import json
import torch
from torch.utils.data import Dataset
from transformers import GPT2Tokenizer, GPT2LMHeadModel, AutoModelForCausalLM, AutoTokenizer
from transformers import Trainer, TrainingArguments

class StringEditDataset(Dataset):
    def __init__(self, corpus_path, split='train', max_examples=5):
        with open(corpus_path) as f:
            corpus = json.load(f)
        self.data = []
        for datum in corpus[split]:
            assert datum["re"].startswith("<") and datum["re"].endswith(">")
            search_rule, replace_rule = datum["re"][1:-1].split("@")
            if is_multiple_non_literals(search_rule) or is_multiple_non_literals(replace_rule):
                continue
            self.data.append(datum)
        self.max_examples = max_examples

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        datum = self.data[idx]
        examples = datum["examples"]
        hint = " ".join(datum["hint"][1:-1])
        n_examples = min(self.max_examples, len(examples) - 1)
        input_output_pairs = [
            # f"{' '.join(inp[1:-1])} → {' '.join(out[1:-1])}"
            f"{''.join(inp[1:-1])} → {''.join(out[1:-1])}"
            for inp, out in examples[:n_examples]
        ]
        test_inp, test_out = examples[n_examples]
        # x_i = "\n".join(input_output_pairs + [f"{''.join(test_inp[1:-1])} → ?\n"])
        # y_i = ' '.join(test_out[1:-1])
        
        x_i = "\n".join(input_output_pairs + [f"What is '{''.join(test_inp[1:-1])}' edited to?"])
        y_i = ''.join(test_out[1:-1])
        r_i = hint
        return x_i, r_i, y_i

dataset = StringEditDataset("corpus.json", split='train', max_examples=5)
