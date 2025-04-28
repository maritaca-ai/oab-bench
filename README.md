# OAB-Bench
| [**Paper**](https://arxiv.org/abs/XXXX.XXXXX) | [**Dataset**](https://huggingface.co/datasets/maritaca-ai/oab-bench) |

OAB-Bench is a benchmark for evaluating Large Language Models (LLMs) on legal writing tasks, specifically designed for the Brazilian Bar Examination (OAB). The benchmark comprises 105 questions across seven areas of law from recent editions of the exam.

- OAB-Bench evaluates LLMs on their ability to write legal documents and answer discursive questions
- The benchmark includes comprehensive evaluation guidelines used by human examiners
- Results show that frontier models like Claude-3.5 Sonnet can achieve passing grades (≥6.0) in most exams
- The evaluation pipeline uses LLMs as automated judges, achieving strong correlation with human scores

## News
- [2025/04] 🔥 Paper accepted at ICAIL 2025 (International Conference on Artificial Intelligence and Law)
- [2025/04] Initial release of the benchmark and evaluation pipeline

## Contents
- [Installation](#installation)
- [Usage](#usage)
- [Results](#results)
- [Citation](#citation)

## Installation

The codebase is based on [FastChat](https://github.com/lm-sys/FastChat) and can be installed via pip:

```bash
# Install from GitHub
pip install git+https://github.com/maritaca-ai/oab-bench.git

# Or install from local source
git clone https://github.com/maritaca-ai/oab-bench.git
cd oab-bench
pip install -e .
```

## Usage

The benchmark evaluation pipeline consists of three main scripts:

1. Generate model responses for a specific model:
```bash
python gen_api_answer.py 
    --bench-name oab_bench 
    --model sabia-3-2024-12-11
    --openai-api-base "https://chat.maritaca.ai/api" 
    --openai-key-env MARITACA_API_KEY
    --parallel 10
```

2. Generate automated evaluations using an LLM judge:
```bash
python gen_judgment.py 
    --bench-name oab_bench 
    --judge-model o1-2024-12-17 
    --mode single
    --model-list sabia-3-2024-12-11
    --parallel 10
```

3. Visualize results:
```bash
python show_result.py --bench-name oab_bench --judge-model o1-2024-12-17
```

## Results

Our evaluation of four LLMs on OAB-Bench shows:

| Model | Average Score | Passing Rate | Best Area |
| --- | --- | --- | --- |
| Claude-3.5 Sonnet | 7.93 | 100% | Constitutional Law (8.43) |
| GPT-4o | 6.87 | 86% | Civil Law (7.42) |
| Sabiá-3 | 6.55 | 76% | Labor Law (7.17) |
| Qwen2.5-72B | 5.21 | 24% | Administrative Law (7.00) |

The LLM judge (o1) shows strong correlation with human scores when evaluating approved exams, with Mean Absolute Error (MAE) ranging from 0.04 to 0.28 across different law areas.

## Citation

If you find this work helpful, please cite our paper:

```
@inproceedings{pires2025automatic,
  title={Automatic Legal Writing Evaluation of LLMs},
  author={Pires, Ramon and Malaquias Junior, Roseval and Nogueira, Rodrigo},
  booktitle={Proceedings of the International Conference on Artificial Intelligence and Law (ICAIL)},
  year={2025}
}
```
