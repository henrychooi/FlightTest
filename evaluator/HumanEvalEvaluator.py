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

    def _load(self):

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        # print some ascii art

        print(rf"""
         __________
        /  ___  ___ \       FlightTest: HumanEval Benchmark
        / / @ \/ @ \ \      
        \ \___/\___/ /\     model_path: {self.model_path}
        \____\/____/||      model_type: hf
        /     /\\\\\//      num_samples: {self.num_samples}
        |     |\\\\\\       device: {device}
        \      \\\\\\       data_path: {self.data_path}
        \______/\\\\        output_path: {self.output_path}
            _||_||_         
            -- --           It is strongly recommended to check all parameters above before proceeding.
        """)
        print(rf"""
        ***************************************************************************************
        *                                       WARNING!                                      *
        *                                                                                     *
        *       This program exists to execute untrusted model-generated code. Although       *
        *      it is highly unlikely that model-generated code will do something overtly      *
        *       malicious in response to this test suite, model-generated code may act        *
        *           destructively due to a lack of model capability or alignment.             *
        *                                                                                     *
        *      Users are strongly encouraged to sandbox this evaluation suite so that it      *
        *       does not perform destructive actions on their host or network. For more       *
        *       information on how OpenAI sandboxes its code, see the accompanying paper.     *
        *                                                                                     *
        ***************************************************************************************
        """)

        # sanity checks to not waste time on invalid parameters
        assert self.num_samples > 0, "num_samples must be greater than 0."
        assert os.path.exists(self.model_path), f"Model path {self.model_path} does not exist."
        assert os.path.exists(self.data_path), f"Data path {self.data_path} does not exist."
        assert os.path.exists(self.output_path), f"Output path {self.output_path} does not exist."

        if self.num_samples > 1:
            print("Warning: num_samples is set to more than 1. This will generate multiple samples for each problem. \n")
        
        if self.model_type == 'hf':
  
            tokenizer = AutoTokenizer.from_pretrained(self.model_path)
            # Loads a pretrained model from the specified directory (if exists), otherwise pulls from HuggingFace
            model = AutoModelForCausalLM.from_pretrained(
                self.model_path,
                device_map="auto",
                torch_dtype="bfloat16",
            )
            tokenizer.pad_token = tokenizer.eos_token
            model.eval()

        else:
            raise ValueError("Model type unsupported. Please set --model_type=hf.")
        return device, model, tokenizer

    # Inference pipeline for HuggingFace models
    def _generate_samples(self):

        device, model, tokenizer = self._load()

        candidates = []

        problems = list(stream_jsonl(self.data_path))

        for _, problem in tqdm(enumerate(problems), total=len(problems), desc="Running inference", unit="problem"):
            prompt = problem['prompt']

            # Generate multiple candidate solutions for each problem
            problem_candidates = []

            # Create a progress bar for the inner loop (samples per problem)
            with tqdm(total=self.num_samples, desc="Generating samples", unit="sample", leave=False) as pbar:
                for _ in range(self.num_samples):
                    
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


        sample_file = os.path.join(self.output_path, f"sample_{os.path.basename(os.path.normpath(self.model_path))}_pass@{self.num_samples}.jsonl")
        write_candidates_to_jsonl(candidates, sample_file)
        return sample_file
    
    def evaluate_model(self):
        sample_file = self._generate_samples()
        pass_at_k = evaluate_functional_correctness(sample_file, problem_file=self.data_path)
        print()
        print(pass_at_k)
        return None