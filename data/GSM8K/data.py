from datasets import load_from_disk, load_dataset

def save_dataset(format: str = "jsonl"):
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


def load_gsm8k(format: str = "jsonl"):
    """
    Loads the GSM8K dataset from a specified format.

    Parameters:
    format (str): The format of the dataset to load. Options: "jsonl" or "arrow".

    Returns:
    dataset (Dataset): The loaded dataset.
    """
    if format == "jsonl":
        dataset = load_dataset("json", data_files=r"gsm8k_jsonl/gsm8k_test.jsonl")
        print("Loaded dataset from JSONL.")
    elif format == "arrow":
        dataset = load_from_disk(r"gsm8k_arrow/test")
        print("Loaded dataset from Arrow format.")
    else:
        raise ValueError("Unsupported format. Use 'jsonl' or 'arrow'.")

    return dataset