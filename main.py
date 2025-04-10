from evaluator.MMLUEvaluator import MMLUEvaluator
from evaluator.HumanEvalEvaluator import HumanEvalEvaluator
from evaluator.GSM8KEvaluator import GSM8KEvaluator
from evaluator.ARCChallengeEvaluator import ARCChallengeEvaluator
from typing import Optional
from utils.HumanEval.data import DATASET_DIR as HUMANEVAL_DATASET_DIR
from utils.GSM8K.defaults import (
    DATASET_DIR as GSM8K_DATASET_DIR,
    PROMPT_DIR as GSM8K_PROMPT_DIR,
    MODEL_DIR as GSM8K_MODEL_DIR,
    OUTPUT_DIR as GSM8K_OUTPUT_DIR,
)
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
            llm_evaluator.evaluate_model()
        case "gsm8k":
                is_chat_model_flag = not kwargs.get('base_model', False)

                llm_evaluator = GSM8KEvaluator(
                    model_path=kwargs['model_path'],
                    data_path=kwargs['data_path'],
                    output_path=kwargs['output_path'],
                    prompt_path=kwargs.get('prompt_path'),
                    is_chat_model=is_chat_model_flag,
                    temperature=kwargs.get('temperature', 0.0), 
                    seed=kwargs.get('seed', 42),              
                    n_shot=kwargs.get('n_shot', 8),             
                    max_new_tokens=kwargs.get('max_new_tokens', 512)
                )
                llm_evaluator.evaluate_model()

        case "humaneval":
            # Check if evaluate_only is set
            if kwargs.get('evaluate_only'):
                assert kwargs.get('sample_file') is not None, "Sample file must be provided when evaluate_only is set. Please specify --sample_file PATH_TO_SAMPLE_FILE"
                assert os.path.exists(kwargs['sample_file']), f"Sample file {kwargs['sample_file']} does not exist."

            # If output or data path not specified, set default values
            if 'output_path' not in kwargs or kwargs['output_path'] is None:
                kwargs['output_path'] = os.path.join("output", "HumanEval")
            
            if 'data_path' not in kwargs or kwargs['data_path'] is None:
                kwargs['data_path'] = HUMANEVAL_DATASET_DIR

            # ensure the output directory exists
            os.makedirs(kwargs['output_path'], exist_ok=True)

            llm_evaluator = HumanEvalEvaluator(
                model_path=kwargs['model_path'],
                num_samples=kwargs['num_samples'],
                model_type=kwargs['model_type'],
                data_path=kwargs['data_path'],
                output_path=kwargs['output_path'],
                debug=kwargs['debug'],
                evaluate_only=kwargs.get('evaluate_only', False),
                sample_file=kwargs.get('sample_file', None)
            )
            llm_evaluator.evaluate_model()

        case "arcchallenge":
            evaluator = ARCChallengeEvaluator(
                model_path=kwargs["model_path"], 
                data_path=kwargs['data_path'], 
                output_path=kwargs['output_path']
            )
            evaluator.evaluate_model()  
            
        case "all":
            llm_evaluator = MMLUEvaluator(
                kwargs['model_path'], 
                kwargs['data_path'], 
                kwargs.get('prompt_path'), 
                kwargs['output_path'], 
                kwargs.get('param_size'), 
                "llama", 
                5
            )
            llm_evaluator.evaluate_model()

            is_chat_model_flag = not kwargs.get('base_model', False)

            llm_evaluator = GSM8KEvaluator(
                model_path=kwargs['model_path'],
                data_path=kwargs['data_path'],
                output_path=kwargs['output_path'],
                prompt_path=kwargs.get('prompt_path'),
                is_chat_model=is_chat_model_flag,
                temperature=kwargs.get('temperature', 0.0), 
                seed=kwargs.get('seed', 42),              
                n_shot=kwargs.get('n_shot', 8),             
                max_new_tokens=kwargs.get('max_new_tokens', 512)
            )
            llm_evaluator.evaluate_model()

            # Check if evaluate_only is set
            if kwargs.get('evaluate_only'):
                assert kwargs.get('sample_file') is not None, "Sample file must be provided when evaluate_only is set. Please specify --sample_file PATH_TO_SAMPLE_FILE"
                assert os.path.exists(kwargs['sample_file']), f"Sample file {kwargs['sample_file']} does not exist."

            # If output or data path not specified, set default values
            if 'output_path' not in kwargs or kwargs['output_path'] is None:
                kwargs['output_path'] = os.path.join("output", "HumanEval")
            
            if 'data_path' not in kwargs or kwargs['data_path'] is None:
                kwargs['data_path'] = HUMANEVAL_DATASET_DIR

            # ensure the output directory exists
            os.makedirs(kwargs['output_path'], exist_ok=True)

            llm_evaluator = HumanEvalEvaluator(
                model_path=kwargs['model_path'],
                num_samples=kwargs['num_samples'],
                model_type=kwargs['model_type'],
                data_path=kwargs['data_path'],
                output_path=kwargs['output_path'],
                debug=kwargs['debug'],
                evaluate_only=kwargs.get('evaluate_only', False),
                sample_file=kwargs.get('sample_file', None)
            )
            llm_evaluator.evaluate_model()

            evaluator = ARCChallengeEvaluator(
                model_path=kwargs["model_path"], 
                data_path=kwargs['data_path'], 
                output_path=kwargs['output_path']
            )
            evaluator.evaluate_model()  
        case _: 
            raise ValueError("Error, unsupported type of evaluator")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()    
    parser.add_argument('--model_path', type=str, required=True)
    parser.add_argument('--output_path', type=str, required=False, help="Path to the output directory. Defaults to output/{evaluator}/ .")
    parser.add_argument('--data_path', type=str, required=False, help="Path to the dataset file. If not specified, uses the default dataset.")
    parser.add_argument('--prompt_path', type=str, required=False)
    parser.add_argument('--param_size', type=int, required=False)
    parser.add_argument('--model_type', type=str, required=False, choices=['hf'], default='hf', help="Model type. Currently, only Hugging Face models are supported. Defaults to 'hf'.")
    parser.add_argument('--num_samples', type=int, default=1, required=False, help="Number of samples to generate for the HumanEval benchmark. Defaults to 1 for pass@1 score.")
    parser.add_argument("--evaluate_only", action='store_true', help="Only runs the evaluation without generating samples. Requires a path to a sample file.")
    parser.add_argument('--sample_file', type=str, required=False, help="Path to the sample file for evaluation. Required if --evaluate_only is set.")
    parser.add_argument('--evaluator', type=str, required=True, choices=['mmlu', 'gsm8k', 'humaneval', 'arcchallenge', 'all'])


    parser.add_argument("--base_model", action='store_true', help="Set this flag if evaluating a base model (uses few-shot completion prompt). Default is to assume a chat model.")
    parser.add_argument("--n_shot", type=int, default=8, help="Number of few-shot examples (max 8).")
    parser.add_argument('--temperature', type=float, default=0)
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    args = parser.parse_args()

    kwargs = vars(args)
    evaluator = kwargs.pop('evaluator')
    main(evaluator, **kwargs)