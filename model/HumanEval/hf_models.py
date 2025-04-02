# Inference pipeline for HuggingFace models

from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import re
import json
import os
from utils.HumanEval.extractor import extractor
from prompt.HumanEval.prompts import SYSTEM_PROMPT
from utils.HumanEval.data import stream_jsonl, DATASET_DIR

def generate_samples(k, model_dir, sample_file):

    def _clean_metadata(text):
        return re.sub(r'^\n\nMETADATA = \{.*?\}\n\n\n', '', text, flags=re.DOTALL)

    def _write_candidates_to_jsonl(candidates, output_file):
        with open(os.path.join('data', output_file), 'w', encoding='utf-8') as f:
            for i, candidate_list in enumerate(candidates):
                for comp in candidate_list:
                    entry = {
                        "task_id": f"HumanEval/{i}",
                        "completion": comp
                    }
                    f.write(json.dumps(entry) + '\n')

    print("Loading model...")

    device = "cuda" if torch.cuda.is_available() else "cpu"

    tokenizer = AutoTokenizer.from_pretrained(model_dir)

    # Loads a pretrained model from the specified directory (if exists), otherwise pulls from HuggingFace
    model = AutoModelForCausalLM.from_pretrained(
        model_dir,
        device_map="auto",
        torch_dtype="bfloat16",
    )

    tokenizer.pad_token = tokenizer.eos_token

    model.eval()

    print("Model loaded.")

    # Lists to store test cases and predictions
    test_cases = []
    candidates = []

    problems = list(stream_jsonl(DATASET_DIR))

    print("Generating code solutions...")
    for _, problem in tqdm(enumerate(problems), total=len(problems), desc="Running inference...", unit="problem"):
        prompt = problem['prompt']
        test_code = problem.get('test', '')
        if test_code:
            test_code = _clean_metadata(test_code)
        # Store the test cases
        test_cases.append(test_code)

        # Generate multiple candidate solutions for each problem
        problem_candidates = []

        # Create a progress bar for the inner loop (samples per problem)
        with tqdm(total=k, desc="Sample", unit="sample", leave=False) as pbar:
            for _ in range(k):
                
                # Prepare chat template
                messages = [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": prompt},
                ]
                

                formatted_prompt = tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
                inputs = tokenizer(formatted_prompt, return_tensors="pt", padding=True).to(device)
                attention_mask = inputs['attention_mask']

                with torch.no_grad():
                    outputs = model.generate(inputs['input_ids'],
                                            attention_mask=attention_mask,
                                            max_new_tokens=512,
                                            do_sample=True,
                                            top_p=0.95,
                                            temperature=0.2,
                                            pad_token_id=tokenizer.pad_token_id,
                                            eos_token_id=tokenizer.eos_token_id)
                     
                prompt_length = inputs['input_ids'].shape[1]

                generated_code = tokenizer.decode(outputs[0][prompt_length:], skip_special_tokens=True)
                    
                extracted_code = extractor(generated_code)

                problem_candidates.append(extracted_code)
                pbar.update(1)
            
        # Add the candidates for the current problem
        candidates.append(problem_candidates)

    print("Code generation complete.")
    _write_candidates_to_jsonl(candidates, f"{sample_file}_candidates.jsonl")