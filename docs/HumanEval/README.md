# FlightTest (HumanEval)

This is a short documentation on the HumanEval benchmark. This benchmark consists of 164 Python programming questions aimed to evaluate a model's Python code-writing capabilities.

The model is provided with a function template and a docstring, which includes examples. This is given as a prompt to the model, and the output is executed in a sandbox environment against multiple test cases.

## Dataset

The dataset and code execution scripts were taken from <a href="https://github.com/openai/human-eval">openai/human-eval</a>.

The problems are stored in a JSON lines file `.jsonl` and are formatted as such:

```json
{
  "task_id": "HumanEval/0",
  "prompt": "...",
  "canonical_solution": "",
  "entry_point": ""
}
```

We extract the `task_id` and `prompt`, then generate a response from the model, and write the results to an output file in the following format:

```json
{"task_id": "HumanEval/0", "completion": ""}
{"task_id": "HumanEval/0", "completion": ""}
{"task_id": "HumanEval/0", "completion": ""}
{"task_id": "HumanEval/0", "completion": ""}
{"task_id": "HumanEval/0", "completion": ""}
```

The number of entries with the same `task_id` correspond to the number of passes `k`, for calculating the model's <b>pass@k</b> score.

## Evaluation

The <b>pass@k</b> metric (<a href="https://arxiv.org/abs/2107.03374">Chen et al., 2019</a>) generates $k$ code solutions per problem. In this benchmark, we generate $n\geq k$ solutions per problem and count the number of correct samples $c\leq n$ that pass the unit tests.

Then, we calculate the unbiased estimator:

$$\text{Pass@k}=\mathbb{E} \left(1-\dfrac{\dbinom{n-c}{k}}{\dbinom{n}{k}}\right)$$

where

- $n$ is the number of coding solutions,
- $c$ is the number of passed solutions,
- $\binom{n}{k}$ is the number of $k$ combinations out of $n$,
- $\binom{n-c}{k}$ is the number of $k$ combinations which do not succeed the unit tests, out of $n$
- $\binom{n-c}{k}$ / $\binom{n}{k}$ gives the probability that all $k$ generated solutions fail to pass the unit tests for the given problem
- The complement, 1 - $\binom{n-c}{k}$ / $\binom{n}{k}$ therefore gives the probability of success

Putting the math aside, this means that the greater the value that we have for $n$, the lower variance and the higher the accuracy of a model's <b>pass@k</b> score. Therefore, the <b>pass@1</b> score calculated from only $n=1$ sample might not be the same as compared to if you took $n=100$ samples instead.

As such, there may be a discrepancy between the results obtained and published results for the same model. In the official implementation by OpenAI, $k$ is set to `[1, 10, 100]` by default. When running the benchmark, you can specify the `--num_samples` flag to specify the value for $n$ to be used, which defaults to 1. This means, that for $n=100$, a pass@1, pass@10, and pass@100 score will be generated.

## Usage

Flags:

```
--model_path  Path to the model to be evaluated
--num_samples Value of n in the formula. Number of samples to generate per problem. Defaults to 1.
--data_path   Path to the dataset file. Defaults to the original dataset if not specified, which is stored in data/HumanEval.
--output_path Path to the output directory of the results file. Defaults to data/HumanEval/results.
--model_type  Type of model. Only Hugging Face models are supported. Defaults to "hf".
```

To run the benchmark at default settings (<b>pass@1</b>),

```bash
python main.py --model_path MODEL_PATH --evaluator humaneval
```

## Known Issues

While evaluation uses very little memory, you might see the following error message when the system is running out of RAM. Since this may cause some correct programs to fail, we recommend that you free some memory and try again.

```
malloc: can't allocate region
```

## Citation

```bibtex
@article{chen2021codex,
  title={Evaluating Large Language Models Trained on Code},
  author={Mark Chen and Jerry Tworek and Heewoo Jun and Qiming Yuan and Henrique Ponde de Oliveira Pinto and Jared Kaplan and Harri Edwards and Yuri Burda and Nicholas Joseph and Greg Brockman and Alex Ray and Raul Puri and Gretchen Krueger and Michael Petrov and Heidy Khlaaf and Girish Sastry and Pamela Mishkin and Brooke Chan and Scott Gray and Nick Ryder and Mikhail Pavlov and Alethea Power and Lukasz Kaiser and Mohammad Bavarian and Clemens Winter and Philippe Tillet and Felipe Petroski Such and Dave Cummings and Matthias Plappert and Fotios Chantzis and Elizabeth Barnes and Ariel Herbert-Voss and William Hebgen Guss and Alex Nichol and Alex Paino and Nikolas Tezak and Jie Tang and Igor Babuschkin and Suchir Balaji and Shantanu Jain and William Saunders and Christopher Hesse and Andrew N. Carr and Jan Leike and Josh Achiam and Vedant Misra and Evan Morikawa and Alec Radford and Matthew Knight and Miles Brundage and Mira Murati and Katie Mayer and Peter Welinder and Bob McGrew and Dario Amodei and Sam McCandlish and Ilya Sutskever and Wojciech Zaremba},
  year={2021},
  eprint={2107.03374},
  archivePrefix={arXiv},
  primaryClass={cs.LG}
}
```
