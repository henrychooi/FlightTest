import re
import os
import random
import torch
import json
import transformers
from tqdm import tqdm
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    StoppingCriteria,
    StoppingCriteriaList,
)
from typing import Optional, List, Dict, Any
from utils.GSM8K.data import load_jsonl
from utils.GSM8K.seeding import seed_everything
from evaluator.BaseEvaluator import BaseEvaluator

# --- Constants ---
GROUND_TRUTH_RE = re.compile(r"#### (\-?[0-9\.\,]+)")
MODEL_ANS_RE = re.compile(r"(-?\$?[\d,]+\.?\d*)|(-?\d+)")
# can be modified according to your requirement
GENERATION_STOP_TOKENS = ["Q:", "</s>", "<|im_end|>", "<|start_header_id|>user<|end_header_id|>", "<|eot_id|>"]

# --- Custom Stopping Criteria (Simplified - definition remains the same) ---
class StopSequenceCriteria(StoppingCriteria):
    def __init__(self, tokenizer, stop_sequences, input_length):
        self.tokenizer = tokenizer
        self.stop_sequences = stop_sequences
        self.input_length = input_length

    def __call__(self, input_ids, scores, **kwargs) -> bool:
        generated_ids = input_ids[:, self.input_length:]
        # Optimization: Decode only the last few tokens to check for endswith
        # Adjust the window size based on max stop sequence length
        max_stop_len = max(len(s) for s in self.stop_sequences) if self.stop_sequences else 0
        min_decode_len = min(generated_ids.shape[1], max_stop_len + 5)
        if min_decode_len > 0:
            generated_text = self.tokenizer.decode(generated_ids[0, -min_decode_len:], skip_special_tokens=True)
            for stop_seq in self.stop_sequences:
                if generated_text.endswith(stop_seq):
                    return True
        return False

# --- GSM8K Evaluator Class ---
class GSM8KEvaluator(BaseEvaluator):
    def __init__(self,
                 model_path: str,
                 data_path: str,
                 output_path: str,
                 prompt_path:str, 
                 temperature: float = 0.0,
                 max_new_tokens: int = 512,
                 stop_sequences: List[str] = GENERATION_STOP_TOKENS,
                 n_shot: int = 8,
                 seed: int = 42,
                 is_chat_model: bool = True):

        super().__init__(model_path=model_path, data_path=data_path, output_path=output_path, prompt_path=prompt_path)
        self.temperature = temperature
        self.max_new_tokens = max_new_tokens
        self.stop_sequences = stop_sequences        
        self.n_shot = n_shot
        self.seed = seed
        self.is_chat_model = is_chat_model


        self.model = None
        self.tokenizer = None
        self.dataset = None
        self._few_shot_examples = []

        transformers.logging.set_verbosity_error()

    @staticmethod
    def _get_hardcoded_examples():
         return [
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


    def setup(self):
        print("Setting up environment...")
        seed_everything(self.seed)
        os.makedirs(self.output_path, exist_ok=True)
        print(f"Output directory: {self.output_path}")

    def _load_model_and_tokenizer(self):
        print(f"Loading model from {self.model_path}...")
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
        print(f"Loading data using data path: {self.data_path}")
        test_filename = "gsm8k_test.jsonl"
        test_filepath = os.path.join(self.data_path, test_filename)
        if not os.path.exists(test_filepath):
            print(f"ERROR: Test file not found at {test_filepath}. Cannot proceed.")
            raise FileNotFoundError(f"GSM8K test file not found at {test_filepath}")
        self.dataset = load_jsonl(test_filepath, instruction="question", output="answer")
        print(f"Loaded {len(self.dataset)} examples from {test_filepath}")
        if not self.dataset:
            raise ValueError("Dataset loaded is empty.")
        
    def _load_prompt_examples(self):
        """Loads few-shot examples from prompt_path or uses hardcoded defaults."""
        loaded_successfully = False
        if self.prompt_path and os.path.exists(self.prompt_path):
            try:
                print(f"Loading prompt examples from {self.prompt_path}...")
                loaded_examples = load_jsonl(self.prompt_path)
                if isinstance(loaded_examples, list) and loaded_examples:
                    # Basic validation: check if items are dicts with required keys
                    if all(isinstance(ex, dict) and 'question' in ex and 'answer' in ex for ex in loaded_examples):
                        self._few_shot_examples = loaded_examples
                        print(f"Successfully loaded {len(self._few_shot_examples)} prompt examples.")
                        loaded_successfully = True
                    else:
                        print("ERROR: Prompt file loaded, but examples lack 'question' or 'answer' keys.")
                else:
                    print("ERROR: Prompt file loaded, but it's empty or not a list.")
            except Exception as e:
                print(f"ERROR: Failed to load or parse prompt file {self.prompt_path}: {e}")

        if not loaded_successfully:
            print("Warning: Using hardcoded few-shot examples.")
            self._few_shot_examples = self._get_hardcoded_examples()

        # Adjust number of examples based on n_shot parameter
        if len(self._few_shot_examples) < self.n_shot:
             print(f"Warning: Requested n_shot={self.n_shot}, but only {len(self._few_shot_examples)} examples available. Using all available.")
             self.n_shot = len(self._few_shot_examples)
        elif len(self._few_shot_examples) > self.n_shot:
             print(f"Selecting {self.n_shot} examples from {len(self._few_shot_examples)} available prompt examples.")
             indices = list(range(len(self._few_shot_examples)))
             random.shuffle(indices) # Shuffles based on seed set in setup()
             selected_indices = indices[:self.n_shot]
             self._few_shot_examples = [self._few_shot_examples[i] for i in selected_indices]

    def load_resources(self):
        self._load_model_and_tokenizer()
        self._load_data()
        self._load_prompt_examples()


    def _build_gsm8k_prompt(self, question: str) -> str:
        if self.is_chat_model:
            chat = [
                {
                    "role": "system",
                    "content": """Follow the examples to solve the math problem. Reason step-by-step and end your response with #### followed by the final numerical answer.

                    Q: There are 15 trees in the grove. Grove workers will plant trees in the grove today. After they are done, there will be 21 trees. How many trees did the grove workers plant today?
                    A: There are 15 trees originally. Then there were 21 trees after some more were planted. So there must have been 21 - 15 = 6. #### 6

                    Q: If there are 3 cars in the parking lot and 2 more cars arrive, how many cars are in the parking lot?
                    A: There are originally 3 cars. 2 more cars arrive. 3 + 2 = 5. #### 5

                    Q: Leah had 32 chocolates and her sister had 42. If they ate 35, how many pieces do they have left in total?
                    A: Originally, Leah had 32 chocolates. Her sister had 42. So in total they had 32 + 42 = 74. After eating 35, they had 74 - 35 = 39. #### 39

                    Q: Jason had 20 lollipops. He gave Denny some lollipops. Now Jason has 12 lollipops. How many lollipops did Jason give to Denny?
                    A: Jason started with 20 lollipops. Then he had 12 after giving some to Denny. So he gave Denny 20 - 12 = 8. #### 8

                    Q: Shawn has five toys. For Christmas, he got two toys each from his mom and dad. How many toys does he have now?
                    A: Shawn started with 5 toys. If he got 2 toys each from his mom and dad, then that is 4 more toys. 5 + 4 = 9. #### 9

                    Q: There were nine computers in the server room. Five more computers were installed each day, from monday to thursday. How many computers are now in the server room?
                    A: There were originally 9 computers. For each of 4 days, 5 more computers were added. So 5 * 4 = 20 computers were added. 9 + 20 is 29. #### 29

                    Q: Michael had 58 golf balls. On tuesday, he lost 23 golf balls. On wednesday, he lost 2 more. How many golf balls did he have at the end of wednesday?
                    A: Michael started with 58 golf balls. After losing 23 on tuesday, he had 58 - 23 = 35. After losing 2 more, he had 35 - 2 = 33 golf balls. #### 33

                    Q: Olivia has $23. She bought five bagels for $3 each. How much money does she have left?
                    A: Olivia had 23 dollars. 5 bagels for 3 dollars each will be 5 x 3 = 15 dollars. So she has 23 - 15 dollars left. 23 - 15 is 8. #### 8"""
                },
                {
                    "role": "user",
                    "content": f"Given the following problem, reason and give a final answer to the problem.\nProblem: {question}\nYour response should end with \"The final answer is [answer]\" where [answer] is the response to the problem.\n"
                },
            ]
            prompt_text = self.tokenizer.apply_chat_template(chat, tokenize=False, add_generation_prompt=True)
        
        else:
            demo_text = ""
            for ex in self._few_shot_examples:
                demo_text += f"Q: {ex['question']}\nA: {ex['answer']}\n\n"

            prompt_text = demo_text + f"Q: {question}\nA:"
        return prompt_text

    def _generate_completion(self, prompt_text: str) -> str:
        inputs = self.tokenizer(prompt_text, return_tensors="pt", return_attention_mask=True).to(self.model.device)
        input_len_tokens = inputs.input_ids.shape[1]

        stop_criteria = StopSequenceCriteria(self.tokenizer, self.stop_sequences, input_len_tokens)
        stopping_criteria_list = StoppingCriteriaList([stop_criteria])

        do_sample = self.temperature > 0.0
        gen_kwargs = {
            "input_ids": inputs.input_ids,
            "attention_mask": inputs.attention_mask,
            "max_new_tokens": self.max_new_tokens,
            "pad_token_id": self.tokenizer.pad_token_id,
            "stopping_criteria": stopping_criteria_list,
            "num_return_sequences": 1,
            "do_sample": do_sample,
            **({"temperature": self.temperature, "top_p": 0.95} if do_sample else {}),
        }

        with torch.no_grad():
            output_ids = self.model.generate(**gen_kwargs)

        generated_part_ids = output_ids[0, input_len_tokens:]
        response = self.tokenizer.decode(generated_part_ids, skip_special_tokens=True)

        for stop_seq in self.stop_sequences:
            if response.endswith(stop_seq):
                response = response[:-len(stop_seq)]
                break

        return response.strip()

    @staticmethod
    def _clean_extracted_number(num_str: Optional[str]) -> Optional[str]:
        if num_str is None: 
            return None
        cleaned = num_str.replace(",", "").replace("$", "").strip()
        if cleaned.endswith("."): 
            cleaned = cleaned[:-1]
        if re.fullmatch(r"-?\d+(\.\d+)?", cleaned):
            return cleaned
        else:
            return None

    @staticmethod
    def extract_ground_truth_answer(answer_text: str) -> Optional[str]:
        match = GROUND_TRUTH_RE.search(answer_text)
        if match:
            return GSM8KEvaluator._clean_extracted_number(match.group(1))
        return None

    @staticmethod
    def extract_model_answer(model_completion: str) -> Optional[str]:
        if model_completion is None:
            return None

        if "####" in model_completion:
            potential_ans = model_completion.split("####")[-1]
            cleaned_num = GSM8KEvaluator._clean_extracted_number(potential_ans)
            if cleaned_num is not None:
                return cleaned_num

        potential_matches = MODEL_ANS_RE.findall(model_completion)
        if not potential_matches:
            return None

        for match_tuple in reversed(potential_matches):
            actual_match = next((m for m in match_tuple if m), None)
            if actual_match:
                cleaned_num = GSM8KEvaluator._clean_extracted_number(actual_match)
                if cleaned_num is not None:
                    return cleaned_num
        return None

    @staticmethod
    def is_correct(model_answer: Optional[str], ground_truth_answer: Optional[str]) -> bool:
        if ground_truth_answer is None: 
            return False
        if model_answer is None: 
            return False
        return model_answer == ground_truth_answer

    def _perform_evaluation_loop(self):
        results_log = []
        num_correct = 0
        num_processed = 0

        print(f"\nEvaluating {len(self.dataset)} samples...")
        for i, sample in enumerate(tqdm(self.dataset, desc="Evaluating Samples")):
            question = sample.get("instruction")
            ground_truth_full = sample.get("output")

            if not question or not ground_truth_full:
                continue

            extracted_ground_truth = self.extract_ground_truth_answer(ground_truth_full)
            if extracted_ground_truth is None:
                results_log.append({
                    "index": i, "question": question, "ground_truth_full": ground_truth_full,
                    "extracted_ground_truth": None, "model_completion": "SKIPPED - GT MISSING",
                    "extracted_model_answer": None, "is_correct": False,
                    "error": "Ground truth marker not found or unparsable"
                })
                continue

            num_processed += 1
            prompt_text = self._build_gsm8k_prompt(question)
            model_completion = self._generate_completion(prompt_text)
            extracted_answer = self.extract_model_answer(model_completion)
            correct = self.is_correct(extracted_answer, extracted_ground_truth)
            if correct:
                num_correct += 1

            results_log.append({
                "index": i, "question": question, "ground_truth_full": ground_truth_full,
                "extracted_ground_truth": extracted_ground_truth,
                "model_completion": model_completion, "extracted_model_answer": extracted_answer,
                "is_correct": correct,
            })

        if num_processed == 0:
            print("ERROR: No samples were successfully processed.")
            accuracy = 0.0
        else:
            accuracy = num_correct / num_processed
            print(f"\nEvaluation Finished.")
            print(f"Problems Processed: {num_processed}")
            print(f"Correct Answers: {num_correct}")
            print(f"Accuracy: {accuracy:.4f}")

        return results_log, accuracy

    def save_results(self, results_data: List[Dict], accuracy: float):
        config = {
            "model_name": self.model_name,
            "model_path": self.model_path,
            "data_path": self.data_path,
            "temperature": self.temperature,
            "stop_sequences": self.stop_sequences,
            "seed": self.seed,
            "max_new_tokens": self.max_new_tokens,
        }
        os.makedirs(self.output_path, exist_ok=True)

        results_filepath = os.path.join(self.output_path, f"{self.model_name}_results.jsonl")
        try:
            with open(results_filepath, "w", encoding='utf-8') as f:
                for result in results_data:
                    f.write(json.dumps(result) + "\n")
            print(f"Detailed results saved to {results_filepath}")
        except IOError as e:
            print(f"ERROR: Failed to save detailed results: {e}")

        score_filepath = os.path.join(self.output_path, f"scores_{self.model_name}.txt")
        num_processed = sum(1 for r in results_data if r.get("model_completion") != "SKIPPED - GT MISSING")
        num_correct = sum(r['is_correct'] for r in results_data)
        try:
            with open(score_filepath, "w", encoding='utf-8') as f:
                f.write(f"--- GSM8K Evaluation Summary ---\n")
                f.write(f"Model Name: {config.get('model_name', 'N/A')}\n")
                f.write(f"Model Path: {config.get('model_path', 'N/A')}\n")
                f.write(f"Data Path: {config.get('data_path', 'N/A')}\n")
                f.write(f"Temperature: {config.get('temperature', 'N/A')}\n")
                f.write(f"Max New Tokens: {config.get('max_new_tokens', 'N/A')}\n")
                f.write(f"Stop Sequences: {config.get('stop_sequences', 'N/A')}\n")
                f.write(f"Prompt: Hardcoded 8-shot CoT (via chat template)\n")
                f.write(f"Seed: {config.get('seed', 'N/A')}\n")
                f.write(f"---------------------------------\n")
                f.write(f"Total Problems Evaluated (Attempted): {len(results_data)}\n")
                f.write(f"Problems Processed (Evaluable GT): {num_processed}\n")
                f.write(f"Correct Answers: {num_correct}\n")
                f.write(f"Accuracy (Correct / Processed): {accuracy:.4f}\n")
                f.write(f"---------------------------------\n")
            print(f"Summary scores saved to {score_filepath}")
        except IOError as e:
            print(f"ERROR: Failed to save summary score: {e}")

    def evaluate_model(self):
        print(f"--- Starting GSM8K Evaluation for {self.model_name} ---")
        self.setup()
        self.load_resources()
        results, accuracy = self._perform_evaluation_loop()
        self.save_results(results, accuracy)
        print(f"--- GSM8K Evaluation Complete for {self.model_name} ---")
        return None

# if __name__ == "__main__":
#     evaluator = GSM8KEvaluator(model_path=r"C:\Users\lh\Documents\evaluations\meta-llama-Llama-3_2-3b", data_path=r"data\GSM8K\gsm8k_jsonl", output_path=r"output\GSM8K", prompt_path=r"prompt\GSM8K\gsm8k_prompt.jsonl")
#     evaluator.evaluate_model()