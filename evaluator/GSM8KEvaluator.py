import re
from collections import Counter

import torch
from tqdm import tqdm
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    StoppingCriteria,
    StoppingCriteriaList,
)

from evaluator.BaseEvaluator import BaseEvaluator
from data.GSM8K.data import load_gsm8k, save_gsm8k, save_results
from prompt.prompt import FEW_SHOT_PROMPT

# Constants
GENERATION_STOP_TOKENS = [
    "Q:",
    "</s>",
    "<|im_end|>",
    "<|eot_id|>",
    "<|start_header_id|>user<|end_header_id|>",
]


class SpecificStringStoppingCriteria(StoppingCriteria):
    """Custom stopping criteria for text generation."""
    
    def __init__(self, tokenizer, stop_strings, input_len):
        self.tokenizer = tokenizer
        self.stop_strings = stop_strings
        self.input_len = input_len

    def __call__(self, input_ids, scores, **kwargs):
        current_text = self.tokenizer.decode(input_ids[0], skip_special_tokens=True)[self.input_len:]
        return any(stop_string in current_text for stop_string in self.stop_strings)


class GSM8KEvaluator(BaseEvaluator):
    """Evaluator for the GSM8K math problem dataset."""
    
    def __init__(self, model_path, data_path, prompt_path, output_path, param_size, model_type, ntrain):
        """
        Initialize GSM8K evaluator.
        
        Args:
            model_path: Path to the model
            data_path: Path to the dataset
            prompt_path: Path to the prompt template (optional)
            output_path: Path to save results
            param_size: Model parameter size
            model_type: Type of model (e.g., "llama")
            ntrain: Number of examples for few-shot learning
        """
        super().__init__(model_path, data_path, prompt_path, output_path)
        self.param_size = param_size
        self.model_type = model_type
        self.ntrain = ntrain
        self.model_name = None
        self.tokenizer = None
        self.model = None

    def load_model(self):
        """Load the model and tokenizer."""
        print("Loading model and tokenizer...")
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_path, device_map="auto", torch_dtype=torch.float16
        )
        print("Model and tokenizer loaded.")

    def evaluator(self):
        """Evaluate model performance on the GSM8K dataset."""
        # Load dataset
        dataset, datasize = load_gsm8k(self.data_path, self.format)
        results = []

        for i in tqdm(range(datasize), desc="Evaluating"):
            example = dataset[i]
            result = self._evaluate_single_example(example)
            results.append(result)

        return results

    def _evaluate_single_example(self, example):
        """
        Process a single example and return the evaluation result.
        
        Args:
            example: Single example from the dataset
            
        Returns:
            Dictionary with evaluation results
        """
        # Prepare input
        input_text = FEW_SHOT_PROMPT.format(question=example["question"])
        inputs = self.tokenizer(input_text, return_tensors="pt").to(self.model.device)
        ground_truth_answer = self.extract_ground_truth(example["answer"])

        # Setup stopping criteria
        stop_criteria = SpecificStringStoppingCriteria(
            self.tokenizer, GENERATION_STOP_TOKENS, len(input_text)
        )
        stopping_criteria_list = StoppingCriteriaList([stop_criteria])

        # Generate answers
        model_answers = self._generate_model_answers(inputs, stopping_criteria_list)
        
        # Process results
        numeric_answers = [ma["numeric"] for ma in model_answers]
        filtered_answers = [num for num in numeric_answers if num is not None]
        
        # Get majority answer if available
        majority_answer = None
        if filtered_answers:
            majority_answer = Counter(filtered_answers).most_common(1)[0][0]
        
        correct = (majority_answer == ground_truth_answer) if majority_answer is not None else False

        return {
            "question": example["question"],
            "gold_answer_text": example["answer"],
            "model_answers_text": [ma["text"] for ma in model_answers],
            "extracted_model_answers": numeric_answers,
            "extracted_gold_answer": ground_truth_answer,
            "majority_answer": majority_answer,
            "correct": correct,
        }

    def _generate_model_answers(self, inputs, stopping_criteria_list):
        """
        Generate one or more answers from the model.
        
        Args:
            inputs: Tokenized inputs
            stopping_criteria_list: List of stopping criteria
            
        Returns:
            List of model answers
        """
        model_answers = []
        
        if hasattr(self.args, 'use_majority_vote') and self.args.use_majority_vote:
            # Generate multiple answers for majority voting
            for _ in range(self.args.n_votes):
                model_answers.append(
                    self._generate_single_answer(
                        inputs, 
                        stopping_criteria_list, 
                        temperature=self.args.temp, 
                        do_sample=True
                    )
                )
        else:
            # Generate a single deterministic answer
            model_answers.append(
                self._generate_single_answer(
                    inputs, 
                    stopping_criteria_list
                )
            )
                
        return model_answers

    def _generate_single_answer(self, inputs, stopping_criteria_list, temperature=0.0, do_sample=False):
        """
        Generate a single answer from the model.
        
        Args:
            inputs: Tokenized inputs
            stopping_criteria_list: List of stopping criteria
            temperature: Sampling temperature (default: 0.0)
            do_sample: Whether to use sampling (default: False)
            
        Returns:
            Dictionary with generated text and extracted numeric answer
        """
        generation_kwargs = {
            "max_new_tokens": 512,
            "pad_token_id": self.tokenizer.eos_token_id,
            "stopping_criteria": stopping_criteria_list,
        }
        
        if do_sample:
            generation_kwargs["temperature"] = temperature
            generation_kwargs["do_sample"] = True
            
        with torch.no_grad():
            outputs = self.model.generate(**inputs, **generation_kwargs)
            
        output_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        output_text = output_text.split("A:")[-1].strip()
        model_answer = self.extract_predicted_answer(output_text)
        
        return {"text": output_text, "numeric": model_answer}

    def compute_metrics(self, results):
        """
        Compute evaluation metrics from results.
        
        Args:
            results: List of evaluation results
            
        Returns:
            Accuracy score
        """
        cnt = sum(1 for result in results if result["correct"])
        total = len(results)
        accuracy = cnt / total if total > 0 else 0.0
        print(f"Accuracy: {cnt} / {total} = {accuracy:.4f}")
        return accuracy

    @staticmethod
    def extract_predicted_answer(text):
        """
        Extract the numeric answer from model output.
        
        Args:
            text: Model output text
            
        Returns:
            Extracted numeric answer or None
        """
        regex_pattern = r"(-?[$0-9.,]{2,})|(-?[0-9]+)"
        regexes_to_ignore = [
            ",",
            "\$",
            r"(?s).*#### ",
            r"\.$",
        ]
        matches = re.findall(regex_pattern, text)
        if matches:
            match = matches[-1]
            # If the match is a tuple, get the non-empty element.
            if isinstance(match, tuple):
                match = next(m for m in match if m)
            answer_text = match.strip()
            for regex in regexes_to_ignore:
                answer_text = re.sub(regex, "", answer_text)
            return answer_text
        return None

    @staticmethod
    def extract_ground_truth(text):
        """
        Extract ground truth answer from reference solution.
        
        Args:
            text: Reference solution text
            
        Returns:
            Extracted ground truth answer
        """
        return text.split("####")[-1].strip()