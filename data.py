import requests, re
from typing import Optional
import random

class TaskEvaluator:
    def parse_llm_output(self, output: str):
        raise NotImplementedError

    def format_question(self, question_data: dict):
        raise NotImplementedError

    def format_prompt(self, question: str):
        raise NotImplementedError

class CommonsenseQAEval(TaskEvaluator):
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        self.prompt_eos = self._load_prompt()

    def _load_prompt(self):
        url = "https://raw.githubusercontent.com/ezelikman/STaR/main/commonsenseqa/prompts.txt"
        prompt = requests.get(url).text
        prompt_eos = re.sub(r"\.\n\n", "." + self.tokenizer.eos_token + "\n\n", prompt)
        if not prompt.endswith(self.tokenizer.eos_token):
            prompt_eos = prompt_eos.rstrip() + self.tokenizer.eos_token
        return prompt_eos

    def format_question(self, question_data):
        question = question_data['question']
        choices_text = ""
        for i, choice_label in enumerate(question_data['choices']['label']):
            choice_text = question_data['choices']['text'][i]
            choices_text += f"({choice_label.lower()}) {choice_text}\n"

        return f"Q: {question}\nAnswer Choices:\n{choices_text.strip()}\nA: "

    def format_prompt(self, question_data):
        return f"{self.prompt_eos}\n\n{self.format_question(question_data)}The answer"

    def parse_llm_output(self, generated_text):
        rationale = generated_text
        final_answer = None

        final_answer = re.findall(r"\(([a-e])\)", generated_text, re.IGNORECASE)
        if final_answer:
            final_answer = final_answer[-1].lower()
            if len(final_answer) > 0:
                last_occurrence_index = generated_text.rfind(f"({final_answer})")
                if last_occurrence_index != -1:
                    rationale = generated_text[:last_occurrence_index].strip()
        return rationale, final_answer


class StringEditingEval(TaskEvaluator):
    def __init__(self, tokenizer, include_hint: bool = False):
        self.tokenizer = tokenizer
        self.include_hint = include_hint
        # Load from turk csvs
        #first_set = load_turk_data('./turk_data.csv')
        #second_set = load_turk_data('./turk_data_2.csv')
        #self.string_dataset = first_set + second_set
        self.prompt_eos = self.tokenizer.eos_token #self._load_prompt()

    #def _load_prompt(self):
    #    url = "https://raw.githubusercontent.com/ezelikman/STaR/main/commonsenseqa/prompts.txt"
    #    prompt = requests.get(url).text
    #    prompt_eos = re.sub(r"\.\n\n", "." + self.tokenizer.eos_token + "\n\n", prompt)
    #    if not prompt.endswith(self.tokenizer.eos_token):
    #        prompt_eos = prompt_eos.rstrip() + self.tokenizer.eos_token
    #    return prompt_eos

    def format_prompt(self, example):
        """Construct the prompt including optional hint."""
        examples = "\n".join([f"{b} is edited to {a}{self.prompt_eos}" for b, a in example["example_pairs"]])
        hint = f"\nHint: {example['hint'].rstrip()}" if self.include_hint and example.get("hint") else ""
        string_instruction=f"""Find the rule applied to edit these example pairs. Apply this rule to the word {example['test_query'].rstrip()} and return the answer as a single word. Think carefully and only return a single word."""
        return f"\n{examples}\n{hint}\n{string_instruction}\nAnswer:"

    def format_question(self, example):
        """Not needed for string editing."""
        return None

    def parse_llm_output(self, generated_text):
        """
        Return full output as rationale and the first token/string after 'Answer:' as final answer.
        """
        rationale = generated_text.strip()
        # Assume the answer is the first non-empty line of the output
        lines = [line.strip() for line in generated_text.strip().split('\n') if line.strip()]
        final_answer = lines[0] if lines else None
        return rationale, final_answer