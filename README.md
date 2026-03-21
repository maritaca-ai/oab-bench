# OAB-Bench
| [**Paper**](https://dl.acm.org/doi/10.1145/3769126.3769227) | [**Dataset**](https://huggingface.co/datasets/maritaca-ai/oab-bench) |

OAB-Bench is a benchmark for evaluating Large Language Models (LLMs) on legal writing tasks from the Brazilian Bar Examination (OAB) Phase 2. The benchmark comprises 210 questions across seven areas of law from six editions (39–44) of the exam.

- Evaluates LLMs on their ability to write legal documents and answer discursive questions
- Includes comprehensive evaluation guidelines used by human examiners
- Supports two judge output formats: **structured** (recommended) and non-structured
- The evaluation pipeline uses LLMs as automated judges, achieving strong correlation with human scores

> **Looking for v1?** The original release (105 questions, exams 39–41) is available at tag [`v1.0`](https://github.com/maritaca-ai/oab-bench/tree/v1.0).

## News
- [2026/03] v2 release: added exams 42–44, structured output evaluation, 12 models evaluated
- [2025/04] Paper accepted at ICAIL 2025 (International Conference on Artificial Intelligence and Law)
- [2025/04] Initial release of the benchmark and evaluation pipeline

## Contents
- [Installation](#installation)
- [Usage](#usage)
- [Results](#results)
- [Structured vs Non-Structured Evaluation](#structured-vs-non-structured-evaluation)
- [Evaluation Cost](#evaluation-cost)
- [Structured Output Format](#structured-output-format)
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

The benchmark evaluation pipeline consists of three steps:

### 1. Generate model responses

Sabiá-4:
```bash
python3 -m gen_api_answer \
    --model sabia-4-2026-01-08 \
    --api-base "https://chat.maritaca.ai/api" \
    --api-key "your-api-key-here" \
    --parallel 10
```

GPT-5.2:
```bash
python3 -m gen_api_answer \
    --model gpt-5.2 \
    --api-key "your-openai-key" \
    --parallel 10
```

Gemini-3-pro-preview:
```bash
python3 -m gen_api_answer \
    --model gemini-3-pro-preview \
    --api-base "https://generativelanguage.googleapis.com/v1beta/openai/" \
    --api-key "your-google-key" \
    --parallel 10  # Google models ignore --max-tokens
```

### 2. Generate automated evaluations using an LLM judge

We recommend using GPT-5.2 as the judge model with **structured output** (`--structured`). Judgments are generated with `reasoning_effort="high"`.

**Structured output (recommended):**
```bash
python3 -m gen_judgment \
    --judge-model gpt-5.2 \
    --model-list sabia-4-2026-01-06 \
    --api-base "https://api.openai.com/v1" \
    --api-key "your-openai-key" \
    --parallel 10 \
    --structured
```

Non-structured (legacy):
```bash
python3 -m gen_judgment \
    --judge-model gpt-5.2 \
    --model-list sabia-4-2026-01-06 \
    --api-base "https://api.openai.com/v1" \
    --api-key "your-openai-key" \
    --parallel 10
```

### 3. Visualize results

```bash
python show_result.py --bench-name oab_bench --judge-model gpt-5.2
```

## Results

Evaluation of 12 LLMs on OAB-Bench using GPT-5.2 as judge with structured output:

| Model | Average Score | Passing Rate |
| --- | --- | --- |
| Gemini-3.1-Pro | 9.39 | 42/42 (100%) |
| Claude Opus 4.6 | 8.82 | 42/42 (100%) |
| GPT-5.2 | 8.66 | 42/42 (100%) |
| Claude Sonnet 4.6 | 8.27 | 42/42 (100%) |
| Sabiá-4 | 7.96 | 42/42 (100%) |
| Gemini-3.1-Flash-Lite | 7.60 | 39/42 (93%) |
| Sabiazinho-4 | 6.91 | 34/42 (81%) |
| Sabiá-3.1 | 6.91 | 35/42 (83%) |
| Qwen3.5-397B | 6.69 | 31/42 (74%) |
| GPT-5-Mini | 6.52 | 31/42 (74%) |
| Qwen3.5-35B | 6.17 | 24/42 (57%) |
| Sabiazinho-3 | 6.12 | 26/42 (62%) |

Passing rate indicates the number of exams (out of 42) where the model scored ≥ 6.0. Each exam corresponds to one of the seven areas of law in a given edition.

## Structured vs Non-Structured Evaluation

We recommend the **structured** evaluation format for two reasons:

1. **Auditable**: each scoring item produces a structured JSON object with `item_id`, `item_description`, `analysis`, and `score`, making it easy to inspect and audit individual judgments.
2. **No arithmetic errors**: the total score is computed programmatically by summing item scores, rather than being extracted from free text via regex — eliminating the risk of the judge model making arithmetic mistakes.

Comparative results between the two formats (GPT-5.2 judge):

| Model | Structured | Non-Structured |
| --- | --- | --- |
| Gemini-3.1-Pro | 9.39 | 9.34 |
| Claude Opus 4.6 | 8.82 | 8.80 |
| GPT-5.2 | 8.66 | 8.61 |
| Claude Sonnet 4.6 | 8.27 | 8.23 |
| Sabiá-4 | 7.96 | 7.92 |
| Gemini-3.1-Flash-Lite | 7.60 | 7.57 |
| Sabiazinho-4 | 6.91 | 6.92 |
| Sabiá-3.1 | 6.91 | 6.89 |
| Qwen3.5-397B | 6.69 | 6.67 |
| GPT-5-Mini | 6.52 | 6.50 |
| Qwen3.5-35B | 6.17 | 6.18 |
| Sabiazinho-3 | 6.12 | 6.07 |

| | Structured | Non-Structured |
| --- | --- | --- |
| **Passing rate** | | |
| Gemini-3.1-Pro | 42/42 | 42/42 |
| Claude Opus 4.6 | 42/42 | 42/42 |
| GPT-5.2 | 42/42 | 42/42 |
| Claude Sonnet 4.6 | 42/42 | 42/42 |
| Sabiá-4 | 42/42 | 42/42 |
| Gemini-3.1-Flash-Lite | 39/42 | 39/42 |
| Sabiazinho-4 | 34/42 | 37/42 |
| Sabiá-3.1 | 35/42 | 32/42 |
| Qwen3.5-397B | 31/42 | 31/42 |
| GPT-5-Mini | 31/42 | 27/42 |
| Qwen3.5-35B | 24/42 | 25/42 |
| Sabiazinho-3 | 26/42 | 26/42 |

Scores are nearly identical across formats, with the same model ranking. The structured format is slightly more expensive (~8%) but provides full traceability of the judge's reasoning.

## Evaluation Cost

Cost breakdown for evaluating all 12 models on 210 questions using GPT-5.2 as judge:

| Metric | Structured | Non-Structured |
| --- | --- | --- |
| Total cost | $63.85 | $58.28 |
| Cost per model | $5.32 | $4.86 |
| Prompt tokens | 7,805,993 | 6,936,593 |
| Completion tokens | 3,709,936 | 3,381,311 |

Pricing based on GPT-5.2 rates: $1.75/1M input tokens, $0.175/1M cached input tokens, $14.00/1M output tokens.

## Structured Output Format

When using `--structured`, the judge produces a `JudgmentResult` object (Pydantic) with the following schema:

```python
class ItemEvaluation(BaseModel):
    item_id: str            # Item identifier (e.g., '1', '2', 'A', 'B')
    item_description: str   # Item description from the scoring table
    analysis: str           # Detailed analysis comparing the candidate's answer with the reference
    score: float            # Score assigned to the item

class JudgmentResult(BaseModel):
    items: List[ItemEvaluation]  # List of evaluated items
    total_score: float           # Total score (sum of all item scores)
```

Each question's scoring table items are evaluated independently, and the `total_score` is the sum of all `score` fields — ensuring arithmetic correctness.

## Citation

If you find this work helpful, please cite our paper:

```bibtex
@inproceedings{10.1145/3769126.3769227,
  author = {Pires, Ramon and Malaquias Junior, Roseval and Nogueira, Rodrigo},
  title = {Automatic Legal Writing Evaluation of LLMs},
  year = {2026},
  isbn = {9798400719394},
  publisher = {Association for Computing Machinery},
  address = {New York, NY, USA},
  url = {https://doi.org/10.1145/3769126.3769227},
  doi = {10.1145/3769126.3769227},
  booktitle = {Proceedings of the Twentieth International Conference on Artificial Intelligence and Law},
  pages = {420--424},
  numpages = {5},
  keywords = {Open-ended Tasks, Legal Writing, Automatic Evaluation, Brazilian Bar Exam, LLM Judge, Large Language Models},
  series = {ICAIL '25}
}
```
