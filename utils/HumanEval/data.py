from typing import Iterable, Dict, List
import gzip
import json
import os


ROOT = os.path.dirname(os.path.abspath(__file__))

# The path of the human eval dataset
# This is stored in ../../data/HumanEval/HumanEval.jsonl.gz
DATASET_DIR = os.path.join(ROOT, "..", "..", "data", "HumanEval", "HumanEval.jsonl.gz")

def read_problems(evalset_file: str = DATASET_DIR) -> Dict[str, Dict]:
    return {task["task_id"]: task for task in stream_jsonl(evalset_file)}


def stream_jsonl(filename: str) -> Iterable[Dict]:
    """
    Parses each jsonl line and yields it as a dictionary
    """
    if filename.endswith(".gz"):
        with open(filename, "rb") as gzfp:
            with gzip.open(gzfp, 'rt') as fp:
                for line in fp:
                    if any(not x.isspace() for x in line):
                        yield json.loads(line)
    else:
        with open(filename, "r") as fp:
            for line in fp:
                if any(not x.isspace() for x in line):
                    yield json.loads(line)


def write_jsonl(filename: str, data: Iterable[Dict], append: bool = False):
    """
    Writes an iterable of dictionaries to jsonl
    """
    if append:
        mode = 'ab'
    else:
        mode = 'wb'
    filename = os.path.expanduser(filename)
    if filename.endswith(".gz"):
        with open(filename, mode) as fp:
            with gzip.GzipFile(fileobj=fp, mode='wb') as gzfp:
                for x in data:
                    gzfp.write((json.dumps(x) + "\n").encode('utf-8'))
    else:
        with open(filename, mode) as fp:
            for x in data:
                fp.write((json.dumps(x) + "\n").encode('utf-8'))

def write_candidates_to_jsonl(candidates: List[str], output_file: str):
        """
        Writes a list of candidates to a jsonl file.
        """
        with open(os.path.join('data', output_file), 'w', encoding='utf-8') as f:
            for i, candidate_list in enumerate(candidates):
                for comp in candidate_list:
                    entry = {
                        "task_id": f"HumanEval/{i}",
                        "completion": comp
                    }
                    f.write(json.dumps(entry) + '\n')