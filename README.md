# Reasoning Trace Distillation

Research code for compressing and distilling large-model (teacher) reasoning traces on MATH-500 to train a smaller
student more efficiently. Frame: **identify -> compress -> transfer**.

- Teachers: Qwen2.5-7B-Instruct; DeepSeek-R1-Distill-Qwen-7B (reasoning-tuned comparison)
- Student: Qwen2.5-1.5B-Instruct. Baselines: DeepSeek-R1-Distill-Qwen-1.5B, SmolLM2-1.7B-Instruct
- Benchmark: MATH-500 (`HuggingFaceH4/MATH-500`, test split)

Code only. Trace files, figures and slides are not in this repo.

**Start here:** [`STATUS.md`](STATUS.md) (done / left) - [`RESULTS.md`](RESULTS.md) (every number, with validity status) -
[`GLOSSARY.md`](GLOSSARY.md) (what our project terms mean in code).

## Layout

| Folder | Purpose |
|---|---|
| `common/` | Shared logic: `answer_matching.py` (labelling, matching, cut target), `scoring.py` (student log-prob). Every current script imports from here. |
| `generation/` | Produce traces: instruct teacher (k=5/k=8, local and remote), student pass@k, reasoning teacher v2. |
| `analysis/` | `analyze_traces_v3.py`: relabel correctness, bucket where the model first states its final answer, compute the adaptive cut. Works on both trace schemas. |
| `scoring/` | Does the cut trace still carry signal? `score_cut_v2.py` (current), `partition_heuristic.py` (prefix/middle/suffix baselines), `score_by_difficulty.py` (**rerun needed**, see RESULTS E). |
| `diagnostics/` | Truncation checks, trace filtering, cut inspection, `audit_cuts.py` (manual precision audit). |
| `legacy/` | Superseded scripts, kept for history. Do not use for new results. |

## Setup

```
python -m venv .venv
.venv\Scripts\activate.bat
pip install torch --index-url https://download.pytorch.org/whl/cu121
pip install transformers datasets bitsandbytes
```
A 7B teacher needs 4-bit on an 8GB GPU, or 8-bit on 12GB+. Put trace files in `data/` (git-ignored). Run every command from the repo root.

## Current pipeline (run from repo root)

```
# 1. generate (reasoning teacher v2 shown)
python generation/teacher_trace_generation_reasoning_v2.py ...
python diagnostics/filter_truncated_reasoning.py --in data/traces_reasoning_teacher_k5_v2.jsonl --out data/traces_reasoning_teacher_k5_v2_clean.jsonl

# 2. relabel + position + adaptive cut (gold-free)
python analysis/analyze_traces_v3.py --trace-file data/traces_reasoning_teacher_k5_v2_clean.jsonl --out data/cut_reasoning_v3.jsonl --with-difficulty
python analysis/analyze_traces_v3.py --trace-file data/traces_teacher_k5.jsonl --out data/cut_instruct_v3.jsonl --with-difficulty

# 3. does the cut keep the signal?
python scoring/score_cut_v2.py --cut-file data/cut_reasoning_v3.jsonl --trace-file data/traces_reasoning_teacher_k5_v2_clean.jsonl --out data/score_reasoning.json
python scoring/score_cut_v2.py --cut-file data/cut_instruct_v3.jsonl --trace-file data/traces_teacher_k5.jsonl --out data/score_instruct.json

# 4. check the cut by eye
python diagnostics/inspect_cut_diverse.py --cut-file data/cut_reasoning_v3.jsonl --trace-file data/traces_reasoning_teacher_k5_v2_clean.jsonl --n-per-group 4
python diagnostics/audit_cuts.py --cut-file data/cut_reasoning_v3.jsonl --n 30 --out data/audit_reasoning.csv
```

## Working rules
1. A number is quoted only if `RESULTS.md` has a VALID entry for it.
2. Matching/labelling logic lives only in `common/`. Fix it there.
3. Before trusting a new statistic, open ~10 real examples behind it (`diagnostics/inspect_cut_diverse.py`).
4. Every run gets a `RESULTS.md` line: date, script, input, command, numbers, status.
