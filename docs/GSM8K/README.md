# FlightTest (GSM8K)

This is a short documentation on the GSM8K benchmark. GSM8K (Grade School Math 8K) is a dataset of 8.5K high-quality, linguistically diverse grade school math word problems designed to test a language model's arithmetic reasoning ability.

Each problem consists of a natural language question and a final answer. The benchmark requires the model to parse the question, perform step-by-step reasoning (often multi-step arithmetic), and generate the correct final answer.

## Results
As of 7/4/2025:

### Base Models:
| Model         | Accuracy (%) | Reported Accuracy (%)  | Diff (%) |
|---------------|--------------|------------------------|----------|
| llama3.2:3b   | 29.3         | -                      | -        |


### Instruct Models:
| Model         | Accuracy (%) | Reported Accuracy (%)  | Diff (%) |
|---------------|--------------|------------------------|----------|
| llama3.1:8b   | -            | -                      | -        |
| llama3.2:1b   | 33.7         | –                      | –        |
| llama3.2:3b   | -            | –                      | –        |

With hyperparameters:
```bash
- temperature: `0.0`
- max_new_tokens: `512`
- n_shot: `8`
```

## Dataset

The dataset and evaluation scripts are taken from  
[https://github.com/openai/grade-school-math](https://github.com/openai/grade-school-math)

Each sample is a JSON object in a `.jsonl` file, formatted as:

```json
{
  "question": "A robe takes 2 bolts of blue fiber and half that much white fiber.  How many bolts in total does it take?",
  "answer": "It takes 2\/2=<<2\/2=1>>1 bolt of white fiber\nSo the total amount of fabric is 2+1=<<2+1=3>>3 bolts of fabric\n#### 3"
}
```

## Usage
### Flags:

```bash
--model_path    Path to the model to be evaluated
--num_samples   Number of completions per question (defaults to 1)
--prompt_path   Path to the GSM8K .jsonl prompt file (prompt\GSM8K\gsm8k_prompt.jsonl)
--data_path     Path to the GSM8K .jsonl test file (data\GSM8K\gsm8k_jsonl\gsm8k_test.jsonl)
--output_path   Path to store output completions and the scores

Optional:
--n_shot        Number of few-shot examples (max 8)
--cot_flag      Use CoT in few-shot prompts.
--no_cot_flag   Do not use CoT in few-shot prompts
--temperature   Temperature of the LLM
--seed          Random seed
--debug         Enable detailed debug printing
```

To run the model at default settings:
```bash
cd FlightTest
python main.py --evaluator gsm8k 
```
