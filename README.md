# OAB-Bench
| [**Paper**](https://arxiv.org/abs/2504.21202) | [**Dataset**](https://huggingface.co/datasets/maritaca-ai/oab-bench) |

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

Sabiá-3.1:
```bash
python3 -m gen_api_answer \
    --model sabia-3.1-2025-05-08 \
    --api-base "https://chat.maritaca.ai/api" \
    --api-key "your-api-key-here" \
    --parallel 10
```

GPT-4o:
```bash
python3 -m gen_api_answer \
    --model gpt-4o-2024-08-06 \
    --api-key "your-openai-key" \
    --parallel 10
```

Gemini-2.5-flash:
```bash
python3 -m gen_api_answer \
    --model gemini-2.5-flash \
    --api-base "https://generativelanguage.googleapis.com/v1beta/openai/" \
    --api-key "your-google-key" \
    --parallel 10  # Google models ignore --max-tokens
```

2. Generate automated evaluations using an LLM judge:
```bash
python3 -m gen_judgment \
    --judge-model o1-2024-12-17 \
    --model-list sabia-3-2024-12-11 \
    --api-base "https://api.openai.com/v1" \
    --api-key "your-openai-key" \
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
| gemini-2.5-pro | 9.01 | 100% | Civil Law (9.70) |
| o3 | 8.88 | 100% | Administrative Law (9.60) |
| gemini-2.5-flash | 8.48 | 100% | Criminal Law (9.15) |
| Claude-3.5 Sonnet | 7.93 | 100% | Constitutional Law (8.43) |
| Sabiá-3.1 | 7.10 | 76% | Civil Law (7.88) |
| GPT-4o | 6.87 | 86% | Civil Law (7.42) |
| Sabiá-3 | 6.55 | 71% | Labor Law (7.17) |
| Qwen2.5-72B | 5.21 | 24% | Civil Law (5.48) |

The LLM judge (o1) shows strong correlation with human scores when evaluating approved exams, with Mean Absolute Error (MAE) ranging from 0.04 to 0.28 across different law areas.

### Average scores given by different LLM judges

| Model | o1 judge | o3 judge | gemini-2.5-pro judge |
| --- | --- | --- | --- |
| gemini-2.5-pro | 9.01 | 8.75 | 8.73 |
| o3 | 8.88 | 8.52 | 8.52 |
| gemini-2.5-flash | 8.48 | 8.22 | 8.25 |
| Claude-3.5 Sonnet | 7.93 | 7.70 | 7.57 |
| Sabiá-3.1 | 7.10 | 6.71 | 6.85 |
| GPT-4o | 6.87 | 6.73 | 6.53 |
| Sabiá-3 | 6.55 | 6.36 | 6.02 |
| Qwen2.5-72B | 5.21 | 4.99 | 4.63 |

The table above presents a comparison of scores given by different judges to various language models. It is observed that for a given model, there is a relatively low variation in scores provided by judges o1, o3, and gemini-2.5-pro. Additionally, all three judges produced the same ranking order for the models.

The consistency in scores and model ranking across judges suggests that the evaluation criteria are well-defined and that performance differences among models are clear and recognizable, regardless of the judge. This gives us confidence that the applied methodology yields reliable and valid results.

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
