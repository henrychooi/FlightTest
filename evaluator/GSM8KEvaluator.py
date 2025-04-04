import re
import os
import random
import torch
import json
import asyncio

import transformers
from tqdm import tqdm
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    StoppingCriteria,
    StoppingCriteriaList,
)
from typing import Optional, List
from evaluator import BaseEvaluator
from utils.GSM8K.data import load_jsonl
from utils.GSM8K.seeding import seed_everything

# Set transformers verbosity
transformers.logging.set_verbosity_error()


# --- Constants ---

GROUND_TRUTH_RE = re.compile(r"#### (\-?[0-9\.\,]+)")
MODEL_ANS_RE = re.compile(r"(-?\$?[\d,]+\.?\d*)|(-?\d+)")
# can be modified according to your requirement
GENERATION_STOP_TOKENS = ["Q:", "</s>", "<|im_end|>", "<|start_header_id|>user<|end_header_id|>", "<|eot_id|>"]

# --- Custom Stopping Criteria ---

class StopSequenceCriteria(StoppingCriteria):
    # (StopSequenceCriteria class definition remains the same)
    def __init__(self, tokenizer, stop_sequences, input_length):
        self.tokenizer = tokenizer
        self.stop_sequences = stop_sequences
        self.input_length = input_length

    def __call__(self, input_ids, scores, **kwargs) -> bool:
        generated_ids = input_ids[:, self.input_length:]
        generated_text = self.tokenizer.decode(generated_ids[0], skip_special_tokens=True)
        for stop_seq in self.stop_sequences:
            # Check if the *end* of the generated text contains the stop sequence
            # This is often more reliable than just 'in' to avoid premature stopping
            if generated_text.endswith(stop_seq):
                 # print(f"\nStopping on: '{stop_seq}'") # Optional debug print
                 return True
        return False


class GSM8KEvaluator(BaseEvaluator):
    """
    Evaluator for GSM8K using single-sample evaluation.
    Uses hardcoded few-shot prompts.
    """

    def __init__(self,
                 model_path: os.PathLike,
                 data_path: os.PathLike,
                 output_path: os.PathLike,
                 prompt_path: os.PathLike,

                 # Arguments for single-sample evaluation
                 temperature: float = 0.0, # Default to greedy for single pass
                 stop_sequences: Optional[List[str]] = None,
                 n_shot: int = 8,
                 cot_flag: bool = True,
                 seed: int = 42,
                 debug: bool = False):

        super().__init__(model_path=model_path, data_path=data_path, output_path=output_path, prompt_path=prompt_path)

        # Store specific arguments
        self.temperature = temperature
        self.stop_sequences = GENERATION_STOP_TOKENS
        self.do_sample = self.temperature > 0.0 # Determine sampling need
        self.n_shot = n_shot
        self.cot_flag = cot_flag
        self.seed = seed
        self.debug = debug

        # Internal state
        self.model = None
        self.tokenizer = None
        self.dataset = None
        self.demo_text = ""

        print(f"Initialised GSM8K Evaluator for model: {self.model_name}")
        print(f"Temperature: {self.temperature} (Sampling: {self.do_sample})")
        print(f"Stop Sequences: {self.stop_sequences}")

    # --- Helper Methods ---

    def setup(self):
        """Set up environment (seed, directories)."""
        print("Setting up environment...")
        seed_everything(self.seed)
        os.makedirs(self.output_path, exist_ok=True)
        print(f"Seed set to {self.seed}")
        print(f"Output directory: {self.output_path}")

    def load_resources(self):
        """Load model, tokenizer, dataset, and prepare prompts."""
        self._load_model_and_tokenizer()
        self._load_data()
        self._prepare_demo_text() # Prepare few-shot prompt

    def _load_model_and_tokenizer(self):
        """Load the model and tokenizer."""
        print(f"Loading model from {self.model_path} ...")
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path, trust_remote_code=True)
        model_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        print(f"Using model dtype: {model_dtype}")
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_path, device_map="auto", torch_dtype=model_dtype, trust_remote_code=True
        )
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id if self.tokenizer.eos_token_id is not None else 0

        self.model.eval()
        print("Model and tokenizer loaded.")

    def _load_data(self):
        """Load the GSM8K dataset."""
        print(f"Loading data using data path: {self.data_path}")
        test_filename = "gsm8k_test.jsonl"
        test_filepath = os.path.join(self.data_path, test_filename)
        if not os.path.exists(test_filepath):
            print(f"Test file not found at {test_filepath}. Cannot proceed.")
            raise FileNotFoundError(f"GSM8K test file not found at {test_filepath}")
        self.dataset = load_jsonl(test_filepath, instruction="question", output="answer")
        print(f"Loaded {len(self.dataset)} examples from {test_filepath}")

    def _get_prompt_examples(self):
        """Loads or returns few-shot examples."""
        if self.prompt_path and os.path.exists(self.prompt_path):
            print(f"Loading few-shot examples from {self.prompt_path}")
            try:
                # Expecting a JSON list of {"question": ..., "answer": ...}
                # Answer should ideally contain the CoT chain AND the final answer marker
                prompt_examples = load_jsonl(self.prompt_path)
                # Basic validation
                if not isinstance(prompt_examples, list) or not all("question" in ex and "answer" in ex for ex in prompt_examples):
                    raise ValueError("Prompt file must be a JSON list of objects with 'question' and 'answer' keys.")
                return prompt_examples
            except Exception as e:
                print(f"ERROR: Failed to load or parse prompt file {self.prompt_path}: {e}. Falling back.")

        # --- Fallback Default Examples (if no file provided or loading fails) ---
        # These answers include the CoT reasoning and the final '#### <answer>' marker
        print("Warning: Using hardcoded default few-shot examples.")
        default_examples = [
            {
                "question": "There are 15 trees in the grove. Grove workers will plant trees in the grove today. After they are done, there will be 21 trees. How many trees did the grove workers plant today?",
                "answer": "There are 15 trees originally. Then there were 21 trees after some more were planted. So there must have been 21 - 15 = 6. #### 6"
            },
            {
                "question": "If there are 3 cars in the parking lot and 2 more cars arrive, how many cars are in the parking lot?",
                "answer": "There are originally 3 cars. 2 more cars arrive. 3 + 2 = 5. #### 5"
            },
            {
                "question": "Leah had 32 chocolates and her sister had 42. If they ate 35, how many pieces do they have left in total?",
                "answer": "Originally, Leah had 32 chocolates. Her sister had 42. So in total they had 32 + 42 = 74. After eating 35, they had 74 - 35 = 39. #### 39"
            },
            {
                "question": "Jason had 20 lollipops. He gave Denny some lollipops. Now Jason has 12 lollipops. How many lollipops did Jason give to Denny?",
                "answer": "Jason started with 20 lollipops. Then he had 12 after giving some to Denny. So he gave Denny 20 - 12 = 8. #### 8"
            },
            {
                "question": "Shawn has five toys. For Christmas, he got two toys each from his mom and dad. How many toys does he have now?",
                "answer": "Shawn started with 5 toys. If he got 2 toys each from his mom and dad, then that is 4 more toys. 5 + 4 = 9. #### 9"
            },
            {
                "question": "There were nine computers in the server room. Five more computers were installed each day, from monday to thursday. How many computers are now in the server room?",
                "answer": "There were originally 9 computers. For each of 4 days, 5 more computers were added. So 5 * 4 = 20 computers were added. 9 + 20 is 29. #### 29"
            },
            {
                "question": "Michael had 58 golf balls. On tuesday, he lost 23 golf balls. On wednesday, he lost 2 more. How many golf balls did he have at the end of wednesday?",
                "answer": "Michael started with 58 golf balls. After losing 23 on tuesday, he had 58 - 23 = 35. After losing 2 more, he had 35 - 2 = 33 golf balls. #### 33"
            },
            {
                "question": "Olivia has $23. She bought five bagels for $3 each. How much money does she have left?",
                "answer": "Olivia had 23 dollars. 5 bagels for 3 dollars each will be 5 x 3 = 15 dollars. So she has 23 - 15 dollars left. 23 - 15 is 8. #### 8"
            }
        ]

        # If not using CoT, strip the reasoning part *from the default examples*
        # for the demo text generation later.
        if not self.cot_flag:
            for ex in default_examples:
                if "####" in ex["answer"]:
                    # Keep only the marker and the answer value for non-CoT prompts
                    ex["answer"] = "####" + ex["answer"].split("####")[-1].strip()
        return default_examples
    
    def _prepare_demo_text(self):
        """Creates the few-shot demonstration text from examples."""
        prompt_examples = self._get_prompt_examples()
        if len(prompt_examples) < self.n_shot:
            print(f"Warning: Available prompts ({len(prompt_examples)}) < n_shot ({self.n_shot}). Using all.")
            self.n_shot = len(prompt_examples)
        # (Rest of the logic remains the same)
        indices = list(range(len(prompt_examples)))
        random.shuffle(indices)
        selected_indices = indices[:self.n_shot]
        demo_text = ""
        for i in selected_indices:
            ex = prompt_examples[i]
            q, a = ex["question"], ex["answer"]
            if self.cot_flag: demo_text += f"Q: {q}\nA: {a}\n\n"
            else:
                final_ans_part = a.split("####")[-1].strip()
                demo_text += f"Q: {q}\nA: #### {final_ans_part}\n\n"
        self.demo_text = demo_text
        print(f"Prepared {self.n_shot}-shot demo text (CoT: {self.cot_flag}).")


    def _build_prompt(self, question):
        """Builds the full prompt for a given question."""
        return self.demo_text + "Q: " + question + "\nA:"
    

    def _generate_single_sample(self, prompt_text):
        """Generates exactly one completion for a given prompt."""
        inputs = self.tokenizer(prompt_text, return_tensors="pt").to(self.model.device)
        input_len_tokens = inputs.input_ids.shape[1]

        # Stopping criteria uses configured sequences
        stop_criteria = StopSequenceCriteria(self.tokenizer, self.stop_sequences, input_len_tokens)
        stopping_criteria_list = StoppingCriteriaList([stop_criteria])

        gen_kwargs = {
            "input_ids": inputs.input_ids,
            "attention_mask": inputs.attention_mask,
            "max_new_tokens": 512,
            "pad_token_id": self.tokenizer.pad_token_id,
            "stopping_criteria": stopping_criteria_list,
            "num_return_sequences": 1,
            "do_sample": self.do_sample,
        }
        if self.do_sample:
            gen_kwargs["temperature"] = self.temperature
            gen_kwargs["top_p"] = 0.95

        with torch.no_grad():
            output_ids = self.model.generate(**gen_kwargs)

        generated_part_ids = output_ids[0, input_len_tokens:]
        response = self.tokenizer.decode(generated_part_ids, skip_special_tokens=True)

        return response.strip() # Return single string
    
    @staticmethod
    def extract_ground_truth_answer(answer_text):
        """Extracts ground truth """
        match = GROUND_TRUTH_RE.search(answer_text)
        if match: 
            return match.group(1).strip().replace(",", "")
        else: 
            return None

    @staticmethod
    def _clean_extracted_number(num_str):
        """Helper to clean common artifacts from extracted numbers."""
        if num_str is None: 
            return None
        cleaned = num_str.replace(",", "").replace("$", "").strip()
        if cleaned.endswith("."): 
            cleaned = cleaned[:-1]
        if re.fullmatch(r"-?\d+\.?\d*", cleaned) or re.fullmatch(r"-?\d+", cleaned):
             return cleaned
        else: 
            return None

    @classmethod
    def extract_model_answer(cls, model_completion):
        """Extracts the numerical answer from model output"""
        if model_completion is None: 
            return None

        potential_matches = MODEL_ANS_RE.findall(model_completion)
        if not potential_matches:
            return None

        last_number_str = None
        for match_tuple in reversed(potential_matches):
            actual_match = next((m for m in match_tuple if m), None)
            if actual_match:
                 cleaned_num = cls._clean_extracted_number(actual_match)
                 if cleaned_num is not None:
                      last_number_str = cleaned_num
                      break
        return last_number_str


    @staticmethod
    def is_correct(model_answer, ground_truth_answer):
        """Compares answers, handles None."""
        if ground_truth_answer is None or model_answer is None: 
            return False
        return model_answer == ground_truth_answer


    # --- Evaluation Loop (Single Sample) ---

    def _perform_evaluation_loop(self):
        """Generates samples and extracts answers, calculates metrics directly."""
        results_log = []
        num_correct = 0
        num_processed = 0

        print("Evaluating samples...")
        for i, sample in enumerate(tqdm(self.dataset, desc="Evaluating Samples")):
            question = sample["instruction"]
            ground_truth_full = sample["output"]
            extracted_ground_truth = self.extract_ground_truth_answer(ground_truth_full)

            if extracted_ground_truth is None:
                 print(f"Warning: Skipping example {i} due to missing ground truth marker.")
                 continue

            num_processed += 1
            prompt_text = self._build_prompt(question)
            # Generate single sample
            model_completion = self._generate_single_sample(prompt_text)
            # Extract answer 
            extracted_answer = self.extract_model_answer(model_completion)

            # Check whether answer is correct
            correct = self.is_correct(extracted_answer, extracted_ground_truth)
            if correct:
                num_correct += 1

            results_log.append({
                "index": i,
                "question": question,
                "ground_truth_full": ground_truth_full,
                "extracted_ground_truth": extracted_ground_truth,
                "model_completion": model_completion,
                "extracted_model_answer": extracted_answer,
                "is_correct": correct,
            })

            if self.debug and i < 3: # Debug print first few raw results
                 print("-" * 20); print(f"--- Debug Example {i} ---")
                 print(f"Q: {question}"); print(f"GT: {extracted_ground_truth}")
                 print(f"Completion: {model_completion}")
                 print(f"Extracted Ans: {extracted_answer}"); print(f"Correct: {correct}"); print("-" * 20)

        accuracy = (num_correct / num_processed) if num_processed > 0 else 0.0
        print(f"Evaluation Finished. Problems Processed: {num_processed}, Correct: {num_correct}, Accuracy: {accuracy:.4f}")
        return results_log, accuracy


    def save_results(self, results_data, accuracy):
        """Saves detailed results and summary score."""
        
        results_path = os.path.join(self.output_path, f"{self.model_name}_results.jsonl")
        try:
            with open(results_path, "w", encoding='utf-8') as f:
                for result in results_data: f.write(json.dumps(result) + "\n")
            print(f"Detailed results saved to {results_path}")
        except IOError as e: print(f"ERROR: Failed to save detailed results: {e}")

        # Save summary score
        score_path = os.path.join(self.output_path, f"scores_{self.model_name}_simple.txt")
        try:
            with open(score_path, "w", encoding='utf-8') as f:
                f.write(f"Model: {self.model_path}\n")
                f.write(f"Temperature: {self.temperature}\n")
                f.write(f"Stop Sequences: {self.stop_sequences}\n")
                f.write(f"N-Shot: {self.n_shot} (Hardcoded Prompts)\n")
                f.write(f"CoT Prompting: {self.cot_flag}\n")
                f.write(f"Total Problems Evaluated: {len(results_data)}\n")
                f.write(f"Correct Answers: {sum(r['is_correct'] for r in results_data)}\n")
                f.write(f"Accuracy: {accuracy:.4f}\n")
            print(f"Summary scores saved to {score_path}")
        except IOError as e: print(f"ERROR: Failed to save summary score: {e}")


    def _run_sync_evaluation(self):
        """Synchronous wrapper: setup -> load -> evaluate -> save."""
        self.setup()
        self.load_resources()
        results, accuracy = self._perform_evaluation_loop()
        self.save_results(results, accuracy)
        return accuracy

    # --- Implementation of the abstract method ---
    
    async def evaluate_model(self):
        """Asynchronously evaluate the model (single sample)."""
        print(f"Starting asynchronous evaluation task for {self.model_name}...")
        accuracy = await asyncio.to_thread(self._run_sync_evaluation)
        print(f"Asynchronous evaluation task completed for {self.model_name}. Accuracy: {accuracy:.4f}")
        return None
    

if __name__ == "__main__":
    evaluator = GSM8KEvaluator(model_path="../meta-llama--Llama-3_2-1B-Instruct", data_path="../data/GSM8K/gsm8k_jsonl", output_path="../output/GSM8K", prompt_path="../prompt/GSM8K/gsm8k_prompt.jsonl")
    asyncio.run(evaluator.evaluate_model())