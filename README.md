# FlightTest

![Python Version](https://img.shields.io/badge/python-3.11.8+-blue.svg)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

> **FlightTest** is an extensible, plug‑and‑play benchmark harness for systematically evaluating Large Language Models (LLMs) on a curated collection of academic and real‑world tasks.

This respository contains the common benchmarks for evaluating Large Language Models (LLMs) across various tasks using standard academic benchmarks, including general knowledge reasoning (MMLU) and other NLP tasks. It includes datasets, evaluation scripts and baseline results. 

## Table of Contents
- [Introduction](#introduction)
- [Features](#features)
- [Benchmarks Included](#benchmarks-included)
- [Installation](#installation)
- [Results](#results)
- [License](#license)

## Introduction

Evaluating the capabilities of LLMs requires standardised benchmarks. This project aims to provide an easy-to-use suite for running evaluations on popular and challenging datasets like MMLU and GSM8K. It supports various models and produces consistent. comparable results.

## ✈️ Why FlightTest
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


## Installation

Clone the respository and install the dependencies:
```bash
git clone https://github.com/henrychooi/FlightTest.git
cd FlightTest
```

## Usage
Use the `--tasks` argument to choose the metric to evaluate.

For specific arguments, refer to the corresponding  `README.me` for each benchmark.

## Results

Results will be stored in outputs folder.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License
This project is licensed under the [MIT License](LICENSE).

## Acknowledgements

FlightTest stands on the shoulders of the original benchmark authors and the open‑source community. Thank you!

