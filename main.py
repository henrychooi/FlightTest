from evaluator.MMLUEvaluator import MMLUEvaluator
from typing import Optional
import argparse

def main(model_path: str, data_path: str, prompt_path: Optional[str], output_path: str, param_size: Optional[int], evaluator: str):
    match evaluator.lower():
        case "mmlu":
            llm_evaluator = MMLUEvaluator(model_path, data_path, prompt_path, output_path, param_size, "llama", 5)
        case "gsm8k":
            # TODO: Fill in
            pass
        case "humaneval":
            # TODO: Fill in
            pass
        case _: 
            raise ValueError("Error, unsupported type of evaluator")
    llm_evaluator.evaluate_model()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, required=True)
    parser.add_argument('--data_path', type=str, required=True)
    parser.add_argument('--prompt_path', type=str, required=False)
    parser.add_argument('--output_path', type=str, required=True)
    parser.add_argument('--param_size', type=int, required=False)
    parser.add_argument('--evaluator', type=str, required=True, choices=['mmlu', 'gsm8k', 'humaneval'])
    args = parser.parse_args()
    
    main(args.model_path, args.data_path, args.prompt_path, args.output_path, args.param_size, args.evaluator)
