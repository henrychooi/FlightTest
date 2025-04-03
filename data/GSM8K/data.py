from datasets import load_from_disk, load_dataset
import os 
import re
import json

def save_gsm8k(format: str = "jsonl"):
    """
    Saves the GSM8K dataset in the specified format.

    Parameters:
    format (str): The format to save the dataset in. Options: "jsonl" or "arrow".
    """
    dataset = load_dataset("gsm8k", "main", split="test")

    if format == "jsonl":
        dataset.to_json("gsm8k_test.jsonl")
        print("Dataset saved as JSONL: gsm8k_test.jsonl")
    elif format == "arrow":
        dataset.save_to_disk("local_gsm8k")
        print("Dataset saved in Arrow format: local_gsm8k/")
    else:
        print("Unsupported format. Use 'jsonl' or 'arrow'.")


def load_gsm8k(path: str, format: str = "jsonl"):
    """
    Loads the GSM8K dataset from a specified format.

    Parameters:
    format (str): The format of the dataset to load. Options: "jsonl" or "arrow".

    Returns:
    Tuple[Dataset, int]: The loaded dataset and its size.
    """
    if format == "jsonl":
        dataset = load_dataset("json", data_files=path, test_split="test")
        print("Loaded dataset from JSONL.")
    elif format == "arrow":
        dataset = load_from_disk(path)
        print("Loaded dataset from Arrow format.")
    else:
        raise ValueError(f"Unsupported format '{format}'. Use 'jsonl' or 'arrow'.")
    
    datasize = len(dataset)

    return dataset, datasize

def save_results(results, model_name, n_votes, temp, use_majority_vote):
    os.makedirs('eval_results/few_shot', exist_ok=True)
    
    # model_name = model_name.split('/')[-1]    
    model_name = re.sub(r'[^\w\-_\.]', '_', model_name.split('/')[-1])
    result_file = f"eval_results/few_shot/{model_name}"
    if use_majority_vote:
        result_file += f"_maj1@{n_votes}_temp{temp}"
    result_file += "_results.json"

    try:
        with open(result_file, 'w') as f:
            json.dump(results, f, indent=4)
        print(f"Results saved to {result_file}")
    except (OSError, json.JSONDecodeError) as e:
        print(f"Error saving results: {e}")