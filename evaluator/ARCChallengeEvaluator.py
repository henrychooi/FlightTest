import os
import torch
import re
import pandas as pd
from tqdm import tqdm

from .BaseEvaluator import BaseEvaluator
from transformers import AutoModelForCausalLM, AutoTokenizer
from utils.ARCChallenge.utils import create_prompt_list, read_parquet_file


class ARCChallengeEvaluator(BaseEvaluator):

    def __init__(self, model_path, data_path, output_path):
        super().__init__(model_path, data_path, output_path)

    def load_model(self):
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

    def load_df(self):
        df = pd.read_parquet_file(self.data_path, engine='pyarrow')
        df['answerKey'] = df['answerKey'].astype('string')
        return df
    
    def prep_df(self):
        df = self.load_df()
        prompt_list = []
    prompt_template = 'Given the following question and four candidate answers (A, B, C and D), choose the best answer.\nQuestion: {{question.strip()}}\nA. {{choices.text[0]}}\nB. {{choices.text[1]}}\nC. {{choices.text[2]}}{% if choices.text|length > 3 %}\nD. {{choices.text[3]}}{% endif %}\nYour response should end with "The best answer is [the_answer_letter]" where the [the_answer_letter] is one of A, B, C or D.'
    template = Template(prompt_template)
    for _, row in df.iterrows():
        rendered_prompt = template.render(question=row["question"], choices=row["choices"])
        prompt_list.append(rendered_prompt)
    return prompt_list

    def generate_answers(self):

        device, model, tokenizer = self._load()

        prompts_df = create_prompt_list(self.data_path)

        answer_list = []

        for _, prompt in tqdm(prompts_df.iterrows()):

            formatted_prompt = tokenizer.apply_chat_template(prompt['prompt'], add_generation_prompt=True, tokenize=False)
            inputs = tokenizer(formatted_prompt, return_tensors="pt", padding=True).to(device)
            attention_mask = inputs['attention_mask']

            with torch.no_grad():
                outputs = model.generate(inputs['input_ids'],
                                        attention_mask=attention_mask,
                                        max_new_tokens=128,
                                        do_sample=True,
                                        pad_token_id=tokenizer.pad_token_id,
                                        eos_token_id=tokenizer.eos_token_id,
                                        return_full_text=False)
                
            prompt_length = inputs['input_ids'].shape[1]

            answer = tokenizer.decode(outputs[0][prompt_length:], skip_special_tokens=True)
            
            answer_list.append(answer)
                


        sample_file = os.path.join(self.output_path, f"sample_{os.path.basename(os.path.normpath(self.model_path))}_pass@{self.num_shot}.jsonl")
        write_candidates_to_jsonl(candidates, sample_file)
        return sample_file
    
    def evaluate_answer(answer_df):
        for answer in answer_df:
            match = re.search(r'####\s*([A-Da-d1-4])', answer)
            if match:
                first_char = match.group(1)
                print("First alphanumeric character:", first_char)
            else:
                print("No match found.")