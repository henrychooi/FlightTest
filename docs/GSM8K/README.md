# FlightTest (GSM8K)

## Overview
GSM8K (Grade School Math 8K) is a dataset containing 8,500 high-quality, linguistically diverse grade school math word problems. This benchmark evaluates a language model's arithmetic reasoning capabilities, requiring models to:
- Parse natural language questions
- Perform multi-step reasoning
- Execute arithmetic operations
- Generate accurate final answers

Each problem consists of a natural language question and a corresponding final answer.

## Results
*As of 09/04/2025:*

### Base Models:
| Model | Accuracy (%) | Reported Accuracy (%) | Diff (%) |
|---------------|--------------|------------------------|----------|
| llama3.2:3b | 29.3 | - | - |

### Instruct Models:
| Model | Accuracy (%) | Reported Accuracy (%) | Diff (%) |
|---------------|--------------|------------------------|----------|
| llama3.1:8b | - | - | - |
| llama3.2:1b | 39.9 | – | – |
| llama3.2:3b | - | – | – |

### Default Hyperparameters:
```
- temperature: 0.0
- max_new_tokens: 512
- n_shot: 8
```

## Dataset

The GSM8K dataset and evaluation scripts are sourced from [OpenAI's Grade School Math repository](https://github.com/openai/grade-school-math).

### Dataset Format
Each sample is a JSON object in a `.jsonl` file:

```json
{
  "question": "A robe takes 2 bolts of blue fiber and half that much white fiber. How many bolts in total does it take?",
  "answer": "It takes 2/2=<<2/2=1>>1 bolt of white fiber\nSo the total amount of fabric is 2+1=<<2+1=3>>3 bolts of fabric\n#### 3"
}
```

### Dataset Structure
- **Training set**: 7,473 problems
- **Test set**: 1,319 problems
- **Format**: JSONL

## Usage

### Basic Usage
To run the evaluation with default settings:

```bash
cd FlightTest
python main.py --evaluator gsm8k
```

### Advanced Usage
Configure the evaluation using the following command-line arguments:

```bash
python main.py --evaluator gsm8k [options]
```

### Command-Line Arguments:

| Argument | Description | Default |
|----------|-------------|---------|
| `--model_path` | Path to the model to be evaluated | *[default model]* |
| `--num_samples` | Number of completions per question | 1 |
| `--prompt_path` | Path to GSM8K prompt file | `prompt/GSM8K/gsm8k_prompt.jsonl` |
| `--data_path` | Path to GSM8K test file | `data/GSM8K/gsm8k_jsonl/gsm8k_test.jsonl` |
| `--output_path` | Path to store output completions and scores | `output/gsm8k_results.json` |
| `--n_shot` | Number of few-shot examples (max 8) | 8 |
| `--cot_flag` | Use Chain-of-Thought in few-shot prompts | *[not set]* |
| `--no_cot_flag` | Do not use Chain-of-Thought in few-shot prompts | *[not set]* |
| `--temperature` | Temperature of the LLM | 0.0 |
| `--seed` | Random seed | 42 |
| `--debug` | Enable detailed debug printing | *[not set]* |

## Evaluation Metric

The primary metric used for evaluation is `em_maj1@1` (Exact Match Majority at 1), which measures the percentage of problems for which the model produces the correct final answer.
- **Exact Match**: The model's final answer must exactly match the correct answer (not just be semantically equivalent)
- **Majority at 1**: Take the most common answer when multiple samples are generated for each question (though in the default configuration with num_samples=1, this is simply the single answer provided)

## Example Prompt

Here's an example of how a few-shot prompt might be structured:

```
Q: Natalia sold clips to 48 of her friends in April, and then she sold half as many clips in May. How many clips did Natalia sell altogether in April and May?

A: In April, Natalia sold 48 clips.
In May, Natalia sold 48/2 = 24 clips.
So Natalia sold 48 + 24 = 72 clips altogether in April and May.
The answer is 72.
```

## Troubleshooting

Common issues and solutions:

- **Issue**: Model fails to load
  - **Solution**: Verify the model path and ensure enough GPU memory is available

- **Issue**: Out of memory errors
  - **Solution**: Reduce batch size or use a smaller model

- **Issue**: Unexpected output format
  - **Solution**: Check that the prompt formatting matches what the model expects


## Citation

```bibtex
@misc{cobbe2021training,
      title={Training Verifiers to Solve Math Word Problems},
      author={Karl Cobbe and Vineet Kosaraju and Mohammad Bavarian and Jacob Hilton and Reiichiro Nakano and Christopher Hesse and John Schulman},
      year={2021},
      eprint={2110.14168},
      archivePrefix={arXiv},
      primaryClass={cs.LG}
}
```

## License

This project is licensed under the [MIT License](LICENSE).
