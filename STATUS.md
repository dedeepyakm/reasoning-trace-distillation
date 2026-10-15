# Status and handoff (updated 2026-10-15)

Frame: **identify -> compress -> transfer** on MATH-500. Teachers: Qwen2.5-7B-Instruct and DeepSeek-R1-Distill-Qwen-7B. Student: Qwen2.5-1.5B-Instruct.
Everything so far measures a **proxy** (student log-prob of the gold answer given a trace prefix). No student has been fine-tuned on compressed
traces in this repo yet.

## Done
- Trace generation: instruct teacher k=5 (50 -> 100 problems), k=8 extension (50 problems), student pass@1..8, reasoning teacher v2 (232 clean traces, 50 problems).
- Truncation diagnosis/fix for the reasoning teacher (3072 -> 8192 tokens, exact `hit_token_limit`).
- Measurement audit (Oct 14-15): relabelling, boundary-aware and cue-aware matching, gold-free cut, memory-safe scoring. See RESULTS.md.
- First valid comparison (RESULTS.md B-D): reasoning-teacher traces are far more compressible (56% of words kept) and cut traces keep most of the signal (92-94%) for both teachers.

## Left, in this order
**Integrity (small):**
1. Audit instruct-teacher cuts the same way (30 cuts) - RESULTS.md C covers reasoning only.
2. Token counts (not words) for full vs cut, both teachers, to state a real cost saving.
3. Rerun `scoring/score_by_difficulty.py` with current cuts (it backs the old, now-unsupported "beats fixed-fraction" claim).
4. Instruct teacher on all 100 problems with the current pipeline; reasoning teacher on more problems and the 18 dropped rows regenerated at 16k tokens (Phoenix). Only 14-16 reasoning incorrect traces exist, hence the wide interval.

**Research:**
5. **Transfer test (proposal, not yet agreed with supervisors):** fine-tune the student on (a) full traces, (b) adaptive-cut traces, (c) random cut of equal length; evaluate accuracy on problems NOT used for traces
   (e.g. generate training traces from MATH train problems, evaluate on MATH-500). Second axis: instruct vs reasoning teacher. This turns the proxy into a result.
6. Curriculum ordering by pass-rate (Approach C) vs random order; then the bandit-style adaptive curriculum prototype.
7. Cross-family checks (Gemma now; Llama after licence acceptance). Full-precision Phoenix reruns with matched k and 16k-32k budgets (all teacher results so far are 4/8-bit).
8. Meta-learning survey (carried over from earlier meetings).

## How to resume (Windows, from the repo root)
Setup: `python -m venv .venv`, `.venv\Scripts\activate.bat`, `pip install torch --index-url https://download.pytorch.org/whl/cu121`, `pip install transformers datasets bitsandbytes`.
Hardware used: RTX 5070 8GB (laptop). 7B teachers need 4-bit locally. A very long trace can run the GPU out of memory in scoring; the scorer skips and reports such pairs.

Data lives in `data/` (git-ignored, **not in the repo**). Files needed to reproduce RESULTS.md:
`traces_teacher_k5.jsonl` (instruct, 50 problems x5; the 100-problem versions are `traces_teacher_k5_combined.jsonl`), `traces_reasoning_teacher_k5_v2.jsonl` (raw) and
`traces_reasoning_teacher_k5_v2_clean.jsonl` (18 truncated rows dropped). Keep a copy on the shared drive / Phoenix; regenerate with `generation/` scripts if lost.

Reproduce the final numbers:
```
python analysis/analyze_traces_v3.py --trace-file data/traces_reasoning_teacher_k5_v2_clean.jsonl --out data/cut_reasoning_final.jsonl --with-difficulty
python analysis/analyze_traces_v3.py --trace-file data/traces_teacher_k5.jsonl --out data/cut_instruct_final.jsonl --with-difficulty
python scoring/score_cut_v2.py --cut-file data/cut_reasoning_final.jsonl --trace-file data/traces_reasoning_teacher_k5_v2_clean.jsonl --out data/score_reasoning_final.json
python scoring/score_cut_v2.py --cut-file data/cut_instruct_final.jsonl --trace-file data/traces_teacher_k5.jsonl --out data/score_instruct_final.json
python diagnostics/show_relabelled.py --trace-file data/traces_teacher_k5.jsonl --cut-file data/cut_instruct_final.jsonl
python diagnostics/audit_cuts.py --cut-file data/cut_reasoning_final.jsonl --n 30 --out data/audit_reasoning.csv
```
Working rules: see README.md (quote only VALID numbers; matching logic only in `common/`; open ~10 real examples behind any new statistic).

## Open questions
- Is "first statement of the final answer" the right commitment point for a backtracking reasoner? (Audit C: 90% clean, 10% accidental.)
- Does the signal-retention proxy predict real fine-tuning gains? Only item 5 answers this.
