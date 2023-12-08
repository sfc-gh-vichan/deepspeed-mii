import csv

import numpy as np
from tqdm import tqdm
from transformers import AutoTokenizer


class PromptsGenerator:
    def __init__(self, tokenizer_path):
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)

        self.texts = []
        with open('./arxiv.csv', newline='') as file:
            reader = csv.reader(file)
            for row in reader:
                self.texts.append(row[0])
        self.prompt_index = 0

        prompt_template = '''[INST]<<SYS>>
Please summarize the text that is given. Return just the summary and no additional conversational dialog such as ""Sure, here is the summary of the text:"".
<</SYS>>  [/INST]'''
        self.prompt_template_length = len(self.tokenizer.encode(prompt_template))
        np.random.seed(37)

    def generate(self, average_token, variance, max_token, n, show_progress=False):
        if n <= 0:
            return []
        prompts = []
        for i in tqdm(range(n), disable=not show_progress, desc="Generating prompts"):
            prompt_length = min(int(np.random.normal(average_token, variance)), max_token)
            prompt_length = max(prompt_length-self.prompt_template_length, 16)  # avoid prompt too short.
            prompt = self.texts[self.prompt_index]
            self.prompt_index += 1
            if self.prompt_index >= len(self.texts):
                self.prompt_index = 0
            prompt_tokens = self.tokenizer.encode(prompt)[:prompt_length]
            prompt = self.tokenizer.decode(prompt_tokens, skip_special_tokens=True)
            prompt = f'''[INST]<<SYS>>
Please summarize the text that is given. Return just the summary and no additional conversational dialog such as ""Sure, here is the summary of the text:"".
<</SYS>> {prompt} [/INST]'''
            prompts.append(prompt)
        return prompts

    def reset(self):
        self.prompt_index = 0


if __name__ == "__main__":
    tokenizer = AutoTokenizer.from_pretrained('/models/llama-2-7b-chat-hf')
    pg = PromptsGenerator('/models/llama-2-7b-chat-hf')
    prompts = pg.generate(1024, 1024*0.3, 4096-1024, 1024, show_progress=False)

    l = []
    for i in prompts:
        l.append(len(tokenizer.encode(i)))

    print(len(l), max(l), min(l), sum(l)/len(l))
