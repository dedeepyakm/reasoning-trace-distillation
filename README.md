# Reasoning Trace Distillation

Research code for compressing and distilling large-model (teacher) reasoning
traces on MATH-500, to train a smaller student model more efficiently.
Framed as **identify → compress → transfer**: identify which parts of a
teacher's reasoning trace actually matter, compress the trace down to that,
and transfer it to the student.

- **Teacher:** Qwen2.5-7B-Instruct
- **Student:** Qwen2.5-1.5B-Instruct
- **Baselines:** DeepSeek-R1-Distill-Qwen-1.5B, SmolLM2-1.7B-Instruct
- **Benchmark:** [MATH-500](https://huggingface.co/datasets/HuggingFaceH4/MATH-500) (`HuggingFaceH4/MATH-500`, test split)

This repo holds the analysis and generation scripts only. Trace files
(`traces_*.jsonl`) and generated figures/decks are not included — they're
produced locally by running the scripts below in order.

**See [`GLOSSARY.md`](GLOSSARY.md) for precise, code-grounded definitions of
every project-specific term used in our results** — "early/intermediate",
"last two sentences", "never explicit", "correct/incorrect", "adaptive cut",
"signal retention", and the difficulty tiers. These are not standard ML
terms; they're definitions we created for this project, and the glossary
points at the exact function/logic that implements each one.

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate.bat      # Windows cmd
# or: .venv\Scripts\Activate.ps1   (PowerShell)
# or: source .venv/bin/activate    (Linux/Mac)

pip install torch --index-url https://download.pytorch.org/whl/cu121   # match this to your CUDA driver
pip install transformers datasets bitsandbytes
```

A 7B teacher model needs 4-bit quantization to fit on an 8GB GPU, or 8-bit
on 12GB+. The student (1.5B) runs comfortably in bf16 on most GPUs.

## Pipeline, in order

### 1. Generate traces

| Script | What it does |
|---|---|
| `extend_k_experiment_local.py` | Student model: generates k=8 samples/problem on MATH-500, computes pass@1..pass@8. Resumable, runs locally. |
| `extend_k_experiment.py` | Same, for a remote/server environment. |
| `teacher_k8_extension.py` | Extends existing teacher traces from k=5 to k=8 (8-bit quantized), `--existing-only` scopes it to already-covered problems instead of regenerating from scratch. |
| `teacher_trace_generation_local.py` | Generates NEW teacher traces (k=5, 4-bit quantized) for problems not yet covered by an existing trace file — the way we scaled from 50 to 100 problems. Resumable, skips already-covered problems. |

### 2. Analyze where the answer lives in a trace

| Script | What it does |
|---|---|
| `partition_heuristic.py` | Scores student confidence (teacher-forced log-prob of the gold answer) under prefix / middle / suffix truncations of a trace, at various retained fractions. Includes `--strip-last-n` to test whether a "suffix" advantage is just leakage from a concluding sentence. |
| `answer_position_analysis.py` | Pure text analysis (no GPU): finds the first sentence that explicitly states the gold answer, buckets each trace into early/intermediate, last-2-sentences, or never-explicit, and reports accuracy per bucket. |
| `adaptive_cut_analysis.py` | Builds on the above: truncates each trace right after its own first explicit-answer sentence (adaptive, per-trace — not a fixed 50%/75% cut), reports compression achieved. |
| `filter_never_explicit_stats.py` | Recomputes adaptive-cut compression stats excluding never-explicit traces, to isolate the real correct-vs-incorrect effect from the never-explicit confound. |
| `difficulty_stratified_analysis.py` | Same position-bucketing + adaptive-cut logic, split by MATH-500 difficulty level (Easy/Medium/Hard). Also fixes a substring-matching bug in the original scripts above: a numeric gold answer like "2" could falsely match inside "12" or "20" — this version requires a non-digit boundary for numeric answers. |

### 3. Score whether the compressed trace still carries signal

| Script | What it does |
|---|---|
| `score_adaptive_cut.py` | Scores adaptive-cut traces vs. full original traces with the same teacher-forced log-prob method as `partition_heuristic.py`, for direct comparison. |
| `score_by_difficulty.py` | Same scoring, but reruns ALL strategies (prefix/middle/suffix-stripped/adaptive-cut/full) on one consistent dataset, split by Easy/Medium/Hard. Closes a scale mismatch from an earlier run and tests whether adaptive cutting's advantage holds on hard problems specifically. |

### 4. Instruct-tuned vs. reasoning-tuned teacher comparison (in progress)

Every result above used an instruct-tuned teacher (Qwen2.5-7B-Instruct).
This stage tests whether the same position/compression/scoring patterns
hold for a model actually trained to produce long chain-of-thought.

| Script | What it does |
|---|---|
| `teacher_trace_generation_reasoning.py` | **v1** — generates traces from DeepSeek-R1-Distill-Qwen-7B (same scale/family lineage as the instruct teacher), matched protocol (50 problems, k=5). Found a real issue: `max_new_tokens=3072` was too small for this model's long, backtracking style — 29.6% of samples had no `\boxed{}` answer, and direct inspection showed most were cut off mid-sentence, not genuine errors. |
| `check_truncation_reasoning.py` | Diagnostic (no GPU): flags traces that may have been cut off before reaching `\boxed{}`, using trace length as a proxy. (Useful as a first pass, but see `inspect_no_boxed.py` for how we actually confirmed the issue — the length-based proxy alone wasn't reliable here.) |
| `inspect_no_boxed.py` | Diagnostic (no GPU): prints the tail of traces where no `\boxed{}` answer was found, for direct visual confirmation of truncation vs. a genuine extraction-format mismatch. |
| `teacher_trace_generation_reasoning_v2.py` | **v2 — fixed.** Raises `max_new_tokens` to 8192, records an exact `hit_token_limit` flag per sample from the real generated-token count (not a word-count guess), and adds a fallback extractor for answers stated outside `\boxed{}` (e.g. `**Answer:** ...`). This is the version whose output should be used for the actual instruct-vs-reasoning comparison. |

## Typical run, end to end

```bash
# 1. Generate/extend traces (repeat with --existing-file pointing at your running combined file to scale up)
python teacher_trace_generation_local.py --existing-file traces_teacher_k5.jsonl --out traces_teacher_k5_extended.jsonl --num-new-problems 50 --k 5

# merge
Get-Content traces_teacher_k5.jsonl, traces_teacher_k5_extended.jsonl | Set-Content traces_teacher_k5_combined.jsonl

# 2. Position + compression analysis
python answer_position_analysis.py --trace-file traces_teacher_k5_combined.jsonl
python adaptive_cut_analysis.py --trace-file traces_teacher_k5_combined.jsonl --out adaptive_cut_traces_combined.jsonl
python filter_never_explicit_stats.py --cut-file adaptive_cut_traces_combined.jsonl
python difficulty_stratified_analysis.py --trace-file traces_teacher_k5_combined.jsonl --out adaptive_cut_traces_by_difficulty.jsonl

# 3. Signal-retention scoring
python score_adaptive_cut.py --cut-file adaptive_cut_traces_combined.jsonl --trace-file traces_teacher_k5_combined.jsonl
python score_by_difficulty.py --cut-file adaptive_cut_traces_by_difficulty.jsonl --trace-file traces_teacher_k5_combined.jsonl --n-per-tier 25
```

## Key findings so far

- **Position isn't uniform**: only ~49% of teacher traces state the gold answer in the last two sentences; about a quarter state it earlier, mid-trace. The remaining quarter never state it explicitly — and those are correct only ~8% of the time, a strong standalone signal.
- **Adaptive, per-trace cutting beats every fixed-fraction heuristic tested** (prefix, middle, leakage-stripped suffix) at every difficulty tier, and the margin is *largest* on Hard problems — evidence the approach isn't just an easy-question artifact.
- **Teacher accuracy drops sharply with difficulty** (83% Easy, 82% Medium, 47% Hard on the current 100-problem sample), and the "early answer → likely correct" signal weakens accordingly (86% accurate on Easy vs. a coin-flip 50% on Hard) — a difficulty-aware signal, not a constant one.

## Known limitations / open items

- All current teacher-scale results use 4-bit or 8-bit quantization (8GB/12GB consumer GPUs), not full precision.
- Sample sizes are still modest at the Hard/Medium tier for the GPU scoring step (as few as 4-6 examples in some incorrect-trace groups) — treat early difficulty-split numbers as indicative, not final.
- Not yet run: Llama cross-family check (license pending), Gemma cross-family check, bandit-style adaptive curriculum.
- Instruct-vs-reasoning teacher comparison is **in progress** (see pipeline stage 4 above) — v1 surfaced a token-budget truncation bug in the reasoning-teacher generation, now fixed in v2. The position/compression/scoring analysis on the reasoning teacher hasn't been run yet; it needs v2's output, not v1's.
