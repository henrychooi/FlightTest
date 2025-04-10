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

    def __init__(self, model_path, data_path, output_path, n_shot=25, prompt_gen_seed=42, seed=42):
        super().__init__(model_path=model_path, data_path=data_path, prompt_path=None, output_path=output_path)
        self.n_shot = n_shot
        self.prompt_gen_seed = prompt_gen_seed
        self.seed = seed


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

        if self.n_shot > 0:
            n_shot_df = prompt_df.sample(n=self.n_shot, random_state=self.prompt_gen_seed)
            prompt_df = prompt_df.drop(prompt_df.index)

        print(f"Prompt DataFrame created with {len(df)} entries.")
        try:
            print(f"Prompt DataFrame created with {len(n_shot_df)} entries.")
        except:
            n_shot_df = pd.DataFrame([])


        return prompt_df, n_shot_df

    def evaluate_model(self):

        device, model, tokenizer = self._load_model()

        prompt_df, n_shot_df = self._prep_df()

        system_prompt = ''

        response_list = []

        answer_list = []

        ground_truth_list = []

        for _, row in n_shot_df.iterrows():
            system_prompt += row['prompt'] + '\n' + 'The best answer is ' + row['ground_truth'] + '\n\n'
        
        for _, prompt_row in (pbar := tqdm(prompt_df.iterrows(), desc=f"Running inference", total=len(prompt_df))):
            
            if tokenizer.chat_template is None:
                input_string = system_prompt + prompt_row['prompt'] + '\n'
                inputs = tokenizer(input_string, return_tensors="pt", padding=True).to(device)
                attention_mask = inputs['attention_mask']

            else:
                messages = [
                    {"role": "system", "content": ''},
                    {"role": "user", "content": ''},
                ] 
                messages[0]['content'] = system_prompt
                messages[1]['content'] = prompt_row['prompt'] + '\n'
                
                formatted_prompt = tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
                inputs = tokenizer(formatted_prompt, return_tensors="pt", padding=True).to(device)
                attention_mask = inputs['attention_mask']
                prompt_length = inputs['input_ids'].shape[1]

            with torch.no_grad():
                outputs = model.generate(inputs['input_ids'],
                                        attention_mask=attention_mask,
                                        max_new_tokens=48,
                                        do_sample=True,
                                        pad_token_id=tokenizer.pad_token_id,
                                        eos_token_id=tokenizer.eos_token_id)
                
            if tokenizer.chat_template is None:
                response = tokenizer.decode(outputs[0], skip_special_tokens=True)

            else:
                # Remove system prompt if chat template is used
                response = tokenizer.decode(outputs[0][prompt_length:], skip_special_tokens=True)
            
            response_list.append(response)
            answers = re.findall(r'The best answer is ([A-Da-d1-4])\.', response)
            print(answers)

            try:
                answer = Counter(answers).most_common(1)[0][0]

            except:
                answer = None

            answer_list.append(answer)
            ground_truth_list.append(prompt_row['ground_truth'])
            # current_accuracy = calculate_accuracy(answer_list, ground_truth_list)

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