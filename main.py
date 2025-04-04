from evaluator.MMLUEvaluator import MMLUEvaluator
from evaluator.HumanEvalEvaluator import HumanEvalEvaluator
from typing import Optional
from utils.HumanEval.data import DATASET_DIR as HUMANEVAL_DATASET_DIR
import argparse
import os

def main(evaluator: str, **kwargs):
    match evaluator.lower():
        case "mmlu":
            llm_evaluator = MMLUEvaluator(
                kwargs['model_path'], 
                kwargs['data_path'], 
                kwargs.get('prompt_path'), 
                kwargs['output_path'], 
                kwargs.get('param_size'), 
                "llama", 
                5
            )
        case "gsm8k":
            # TODO: Fill in
            pass
        case "humaneval":
            if 'output_path' not in kwargs or kwargs['output_path'] is None:
                kwargs['output_path'] = os.path.join("data", "HumanEval", "results")
            
            if 'data_path' not in kwargs or kwargs['data_path'] is None:
                kwargs['data_path'] = HUMANEVAL_DATASET_DIR

            # ensure the output directory exists
            os.makedirs(kwargs['output_path'], exist_ok=True)

            llm_evaluator = HumanEvalEvaluator(
                model_path=kwargs['model_path'],
                num_samples=kwargs['num_samples'],
                model_type=kwargs['model_type'],
                data_path=kwargs.get('data_path'),
                output_path=kwargs['output_path']
            )

        case _: 
            raise ValueError("Error, unsupported type of evaluator")
    llm_evaluator.evaluate_model()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()    
    parser.add_argument('--model_path', type=str, required=True)
    parser.add_argument('--output_path', type=str, required=False, help="Path to the output directory. Defaults to output/{evaluator}/ .")
    parser.add_argument('--data_path', type=str, required=False, help="Path to the dataset file. If not specified, uses the default dataset.")
    parser.add_argument('--prompt_path', type=str, required=False)
    parser.add_argument('--param_size', type=int, required=False)
    parser.add_argument('--model_type', type=str, required=False, choices=['hf'], default='hf', help="Model type. Currently, only Hugging Face models are supported. Defaults to 'hf'.")
    parser.add_argument('--num_samples', type=int, default=1, required=False, help="Number of samples to generate for the HumanEval benchmark. Defaults to 1 for pass@1 score.")
    parser.add_argument('--evaluator', type=str, required=True, choices=['mmlu', 'gsm8k', 'humaneval'])
    args = parser.parse_args()

    kwargs = vars(args)
    evaluator = kwargs.pop('evaluator')
    main(evaluator, **kwargs)
