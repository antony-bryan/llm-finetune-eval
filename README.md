

# LLM Fine-Tuning + Evaluation Pipeline

Fine-tuning `microsoft/phi-2` on SQL generation using QLoRA, with a full evaluation harness comparing base vs fine-tuned performance.

**Result: 2% → 76% exact match (+74 percentage points)**

## Project Overview

| Item | Detail |
|---|---|
| Base model | [microsoft/phi-2](https://huggingface.co/microsoft/phi-2) |
| Dataset | [b-mc2/sql-create-context](https://huggingface.co/datasets/b-mc2/sql-create-context) |
| Method | QLoRA (4-bit NF4 + LoRA r=16) |
| Hardware | Kaggle T4 x2 (free tier) |
| Fine-tuned adapter (Run 1, lr=2e-4) | [antony-bryan-3D2Y/phi2-sql-lora-lr2e4](https://huggingface.co/antony-bryan-3D2Y/phi2-sql-lora-lr2e4) |
| Fine-tuned adapter (Run 2, lr=5e-4) | [antony-bryan-3D2Y/phi2-sql-lora-lr5e4](https://huggingface.co/antony-bryan-3D2Y/phi2-sql-lora-lr5e4) |
| W&B training runs | [phi2-sql-finetune](https://wandb.ai/antonybryan2-00-anthropic/phi2-sql-finetune) |

## Results

![Results Chart](results/charts/results_comparison.png)

| Model | Exact Match | ROUGE-L | Δ vs Base | Adapter |
|---|---|---|---|---|
| Phi-2 Base | 2.0% | 0.886 | — | — |
| Phi-2 + LoRA Run 1 (lr=2e-4) | **76.0%** | **0.9903** | **+74pp** | [phi2-sql-lora-lr2e4](https://huggingface.co/antony-bryan-3D2Y/phi2-sql-lora-lr2e4) |
| Phi-2 + LoRA Run 2 (lr=5e-4) | 70.0% | 0.9825 | +68pp | [phi2-sql-lora-lr5e4](https://huggingface.co/antony-bryan-3D2Y/phi2-sql-lora-lr5e4) |

Evaluated on 50 held-out samples from sql-create-context (seed=42).

## Qualitative Analysis

Fine-tuning produced zero regressions — every query the base model answered correctly was also answered correctly by the fine-tuned model. Key improvements:

- **Column selection:** Model learned to select only the requested column (`SELECT points FROM`) instead of `SELECT *`
- **String quoting:** Consistent double-quote convention matching training data, vs base model's inconsistent single quotes
- **Multi-condition WHERE:** Correctly handles complex AND conditions across multiple columns
- **No hallucination:** Base model frequently appended lengthy natural language explanations after the SQL query. The fine-tuned model stops cleanly at the SQL statement every time.

**Example:**

> **Question:** Who is the h.h. principal with Jim Haught as h.s. principal?
>
> **Base:** `SELECT hh_principal, wr_principal, hs_principal, maplemere_principal`
>
> **Fine-tuned:** `SELECT hh_principal FROM table_name_55 WHERE hs_principal = "jim haught" ...`
>
> **Ground truth:** `SELECT hh_principal FROM table_name_55 WHERE hs_principal = "jim haught" ...`

## Repo Structure

```
llm-finetune-eval/
├── README.md
├── requirements.txt
├── notebooks/
│   └── phi2-sql-finetune.ipynb
├── results/
│   ├── baseline_results.csv
│   ├── baseline_test_samples.csv
│   ├── predictions_base.csv
│   ├── predictions_run1.csv
│   ├── predictions_run2.csv
│   ├── results_summary.csv
│   └── charts/
│       └── results_comparison.png
```

## Training Details

| Parameter | Value |
|---|---|
| LoRA rank | 16 |
| LoRA alpha | 32 |
| Target modules | q_proj, v_proj |
| Quantization | 4-bit NF4 |
| Training samples | 20,000 |
| Epochs | 2 |
| Effective batch size | 16 |
| Learning rate (Run 1) | 2e-4 |
| Learning rate (Run 2) | 5e-4 |
| Max sequence length | 256 |
| Training time | ~7 hours per run |
| Optimizer | paged_adamw_8bit |

## Environment

| Library | Version |
|---|---|
| transformers | 5.0.0 |
| peft | 0.18.1 |
| trl | 1.2.0 |
| bitsandbytes | 0.49.2 |

Trained on Kaggle free tier (T4 x2, CUDA 12.8, Python 3.12).



