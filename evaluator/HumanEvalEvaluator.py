import os
import torch
from tqdm import tqdm
from .BaseEvaluator import BaseEvaluator
from transformers import AutoModelForCausalLM, AutoTokenizer
from prompt.HumanEval.prompts import SYSTEM_PROMPT
from utils.HumanEval.extractor import extractor
from utils.HumanEval.data import stream_jsonl, write_candidates_to_jsonl
from utils.HumanEval.eval import evaluate_functional_correctness

class HumanEvalEvaluator(BaseEvaluator):
    """
    Evaluator for the HumanEval dataset
    """

    def __init__(self, model_path, num_samples, model_type, data_path, output_path):
        super().__init__(model_path, data_path, None, output_path)
        self.num_samples = num_samples
        self.model_type = model_type

    def _load(self, model_path, model_type):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        if model_type == 'hf':
            print("Loading model...")
            tokenizer = AutoTokenizer.from_pretrained(model_path)
            # Loads a pretrained model from the specified directory (if exists), otherwise pulls from HuggingFace
            model = AutoModelForCausalLM.from_pretrained(
                model_path,
                device_map="auto",
                torch_dtype="bfloat16",
            )
            tokenizer.pad_token = tokenizer.eos_token
            model.eval()
            print("Model loaded.")
        else:
            raise ValueError("Model type unsupported. Please set --model_type=hf.")
        return device, model, tokenizer

    # Inference pipeline for HuggingFace models
    def _generate_samples(self, num_samples, data_path, output_path):

        device, model, tokenizer = self._load(self.model_path, self.model_type)

        candidates = []

        problems = list(stream_jsonl(data_path))

        print("Generating code solutions...")
        for _, problem in tqdm(enumerate(problems), total=len(problems), desc="Running inference...", unit="problem"):
            prompt = problem['prompt']

            # Generate multiple candidate solutions for each problem
            problem_candidates = []

            # Create a progress bar for the inner loop (samples per problem)
            with tqdm(total=num_samples, desc="Sample", unit="sample", leave=False) as pbar:
                for _ in range(num_samples):
                    
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

        sample_file = os.path.join(output_path, f"sample_{os.path.basename(os.path.normpath(self.model_path))}.jsonl")
        write_candidates_to_jsonl(candidates, sample_file)
        return sample_file
    
    def evaluate_model(self):
        sample_file = self._generate_samples(self.num_samples, self.data_path, self.output_path)
        print("Starting model inference...")
        pass_at_k = evaluate_functional_correctness(sample_file, problem_file=self.data_path)
        print(pass_at_k)
        return None