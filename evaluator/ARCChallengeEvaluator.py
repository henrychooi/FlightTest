import os
import torch
import re
import pandas as pd
from tqdm import tqdm
from jinja2 import Template
import pyarrow
from collections import Counter

from .BaseEvaluator import BaseEvaluator
from transformers import AutoModelForCausalLM, AutoTokenizer
from utils.ARCChallenge.utils import calculate_accuracy


class ARCChallengeEvaluator(BaseEvaluator):

    def __init__(self, model_path, data_path, output_path):
        super().__init__(model_path, data_path, None, output_path)

    def _load_model(self):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
  
        tokenizer = AutoTokenizer.from_pretrained(self.model_path)
        model = AutoModelForCausalLM.from_pretrained(
            self.model_path,
            device_map="auto",
            torch_dtype="bfloat16",
        )
        tokenizer.pad_token = tokenizer.eos_token
        model.eval()

        print(f"Model loaded on {device}")

        return device, model, tokenizer
    
    def _prep_df(self):
        df = pd.read_parquet(os.path.join(self.data_path, "test", "test-00000-of-00001.parquet"), engine='pyarrow')
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

        print(prompt_df.head())

        print(f"Prompt DataFrame created with {len(prompt_df)} entries.")

        return prompt_df

    def evaluate_model(self):

        device, model, tokenizer = self._load_model()

        prompt_df = self._prep_df()

        response_list = []

        answer_list = []

        ground_truth_list = []

        for _, prompt_row in (pbar := tqdm(prompt_df.iterrows(), desc=f"Running inference", total=len(prompt_df))):

            inputs = tokenizer(prompt_row['prompt'], return_tensors="pt", padding=True).to(device)
            attention_mask = inputs['attention_mask']

            with torch.no_grad():
                outputs = model.generate(inputs['input_ids'],
                                        attention_mask=attention_mask,
                                        max_new_tokens=128,
                                        do_sample=True,
                                        pad_token_id=tokenizer.pad_token_id,
                                        eos_token_id=tokenizer.eos_token_id)
                
            prompt_length = inputs['input_ids'].shape[1]

            response = tokenizer.decode(outputs[0][prompt_length:], skip_special_tokens=True)
            
            response_list.append(response)

            answers = re.findall(r'The best answer is ([A-Da-d1-4])\.', response)

            answer = Counter(answers).most_common(1)[0][0]

            answer_list.append(answer)

            ground_truth_list.append(prompt_row['ground_truth'])

            current_accuracy = calculate_accuracy(answer_list, ground_truth_list)

            pbar.set_postfix(acc=f"{current_accuracy:.2%}")

        prompt_df.insert(1, 'answer', answer_list)

        prompt_df.insert(1, 'response', response_list)

        output_path = f'{self.output_path}.parquet'

        prompt_df.to_parquet(output_path, engine='pyarrow', index=False)

        print(f"Final accuracy: {current_accuracy:.2%}")

        print(f"Output saved to {output_path}")

    
if __name__ == "__main__":
    evaluator = ARCChallengeEvaluator(model_path="C:/Users/Augustine/models/meta-llama--Llama-3_2-1B-Instruct", data_path="C:/Users/Augustine/Documents/GitHub/FlightTest/data/ARCChallenge/test/test-00000-of-00001.parquet", output_path="C:/Users/Augustine/Documents/GitHub/FlightTest/output/ARCChallenge/meta-llama--Llama-3_2-1B-Instruct_results")
    evaluator.evaluate_model()  