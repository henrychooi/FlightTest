# FlightTest

![Python Version](https://img.shields.io/badge/python-3.11.8+-blue.svg)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

This respository contains the common benchmarks for evaluating Large Language Models (LLMs) across various tasks using standard academic benchmarks, including general knowledge reasoning (MMLU), mathematical problem-solving (MATH) and other NLP tasks. It includes datasets, evaluation scripts and baseline results. 

## Table of Contents
- [Introduction](#introduction)
- [Features](#features)
- [Benchmarks Included](#benchmarks-included)
- [Installation](#installation)
- [Results](#results)
- [License](#license)

## Introduction

Evaluating the capabilities of LLMs requires standardised benchmarks. This project aims to provide an easy-to-use suite for running evaluations on popular and challenging datasets like MMLU and GSM8K. It supports various models and produces consistent. comparable results.

## Features
*   **Standardised Evaluation Scripts:** Reliable implementations for popular LLM benchmarks.
*   **Local Execution:** Designed to run evaluations on your own infrastructure, supporting local models (e.g., downloaded from Hugging Face).
*   **Configurable:** Easily adjust parameters like few-shot count, target device (CPU/GPU) and model loading arguments.
*   **Clear Reporting:** Outputs results in a structured JSON format and scores in a txt format for straightforward analysis and comparison.

## Benchmarks included

The following benchmarks are currently supported:

| Task Type                         | Benchmark                                      | Description                                                                 | Metric        | Paper                                                                                              |
| :-------------------------------- | :--------------------------------------------- | :-------------------------------------------------------------------------- | :------------ | :------------------------------------------------------------------------------------------------- |
| General Knowledge & Reasoning     | **MMLU** (Massive Multitask Language Understanding) | Measures multitask accuracy across 57 diverse subjects (STEM, humanities, etc.). | Accuracy (5-shot) | [Measuring Massive Multitask Language Understanding](https://arxiv.org/abs/2009.03300)             |
| Mathematical Problem Solving      | **GSM8K** (Grade School Math 8K)               | Measures multi-step mathematical reasoning on grade-school word problems.     | em_maj@1 (Exact Match Majority at 1)     | [Training Verifiers to Solve Math Word Problems](https://arxiv.org/abs/2110.14168)                |
| Coding                            | **HumanEval**                                  | Measures ability to generate functionally correct Python code from docstrings. | Pass@k (Pass@1) | [Evaluating Large Language Models Trained on Code](https://arxiv.org/abs/2107.03374)                |
| Complex Reasoning                 | **ARC Challenge** (AI2 Reasoning Challenge)    | Challenging QA requiring complex reasoning and scientific knowledge.         | Accuracy      | [Think you have Solved QA? Try ARC](https://arxiv.org/abs/1803.05457)                             |
| Multilingual Math Problem Solving | **MGSM** (Multilingual Grade School Math)      | Multilingual version of GSM8K testing math reasoning across 10+ languages.  | em (Exact Match)     | [Crosslingual Generalization of Chain-of-Thought Prompting](https://arxiv.org/abs/2211.01786) |


## Installation

Clone the respository and install the dependencies:
```bash
git clone https://github.com/henrychooi/FlightTest.git
cd FlightTest
```

## Usage
Use the `--tasks` argument to choose the metric to evaluate.

## Results

Results will be stored in outputs folder.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

### License
This project is licensed under the [MIT License](LICENSE).