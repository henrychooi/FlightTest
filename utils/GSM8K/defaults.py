from typing import Iterable, Dict, List
import os


ROOT = os.path.dirname(os.path.abspath(__file__))

# The path of the gsm8k dataset
# This is stored in ../../data/GSM8K/gsm8k_jsonl/gsm8k_test.jsonl
DATASET_DIR = os.path.join(ROOT, "..", "..", "data", "GSM8K", "gsm8k_test", "gsm8k_test.jsonl")


PROMPT_DIR = os.path.join(ROOT, "..", "..", "data", "GSM8K", "prompts")


MODEL_DIR = os.path.join(ROOT, "..", "..", "data", "GSM8K", "models")


OUTPUT_DIR = os.path.join(ROOT, "..", "..", "data", "GSM8K", "output")