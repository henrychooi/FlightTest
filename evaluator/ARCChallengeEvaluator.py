import os
import torch
import re
import pandas as pd
from tqdm import tqdm
from jinja2 import Template
import pyarrow

from .BaseEvaluator import BaseEvaluator
from transformers import AutoModelForCausalLM, AutoTokenizer
from utils.ARCChallenge.utils import create_prompt_list, read_parquet_file


class ARCChallengeEvaluator(BaseEvaluator):

    def __init__(self, model_path, data_path, output_path):
        super().__init__(model_path, data_path, output_path)

    def _load_model(self):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
  
        tokenizer = AutoTokenizer.from_pretrained(self.model_path).to(device)
        model = AutoModelForCausalLM.from_pretrained(
            self.model_path,
            device_map="auto",
            torch_dtype="bfloat16",
        )
        tokenizer.pad_token = tokenizer.eos_token
        model.eval()

        return device, model, tokenizer
    
    def _prep_df(self):
        df = pd.read_parquet_file(self.data_path, engine='pyarrow')
        df['answerKey'] = df['answerKey'].astype('string')
        prompt_list = []
        answer_list = []
        prompt= 'Given the following question and four candidate answers (A, B, C and D), choose the best answer.\nQuestion: {{question.strip()}}\nA. {{choices.text[0]}}\nB. {{choices.text[1]}}\nC. {{choices.text[2]}}{% if choices.text|length > 3 %}\nD. {{choices.text[3]}}{% endif %}\nYour response should end with "The best answer is [the_answer_letter]" where the [the_answer_letter] is one of A, B, C or D.'
        template = Template(prompt)

        for _, row in df.iterrows():
            rendered_prompt = template.render(question=row["question"], choices=row["choices"])
            prompt_list.append(rendered_prompt)
            answer_list.append(row["answerKey"])

        prompt_df = pd.DataFrame(list(zip(prompt_list, answer_list)), columns=['prompt','ground_truth'])
        return prompt_df
    
    def _
        

    def evaluate_answer(answer_df):
        for answer in answer_df:
            match = re.search(r'####\s*([A-Da-d1-4])', answer)
            if match:
                first_char = match.group(1)
                print("First alphanumeric character:", first_char)
            else:
                print("No match found.")

    def _generate_answers(self):

        device, model, tokenizer = self._load_model()

        prompt_df = self._prep_df()

        answer_list = []

        for prompt in (pbar := tqdm(prompt_df.iterrows(), desc=f"Running inference", total=len(prompt_df))):

            inputs = tokenizer(prompt, return_tensors="pt", padding=True).to(device)
            attention_mask = inputs['attention_mask']

            with torch.no_grad():
                outputs = model.generate(inputs['input_ids'],
                                        attention_mask=attention_mask,
                                        max_new_tokens=22,
                                        do_sample=True,
                                        pad_token_id=tokenizer.pad_token_id,
                                        eos_token_id=tokenizer.eos_token_id)
                
            prompt_length = inputs['input_ids'].shape[1]

            answer = tokenizer.decode(outputs[0][prompt_length:], skip_special_tokens=True)
            
            answer_list.append(answer)

            pbar.set_postfix(acc=f"{current_accuracy:.2%}")

        prompt_df.insert(1, 'answer', answer_list)

        prompt_df.to_parquet(f'{self.output_path}.parquet')



        return sample_file
    
    def evaluate_answer(answer_df):
        for answer in answer_df:
            match = re.search(r'####\s*([A-Da-d1-4])', answer)
            if match:
                first_char = match.group(1)
                print("First alphanumeric character:", first_char)
            else:
                print("No match found.")