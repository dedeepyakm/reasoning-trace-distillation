# Status: work done / work left (updated 2026-10-14)

Research frame: **identify -> compress -> transfer** on MATH-500. Teacher Qwen2.5-7B-Instruct (plus DeepSeek-R1-Distill-Qwen-7B
as a reasoning-tuned comparison), student Qwen2.5-1.5B-Instruct. Everything so far measures a **proxy** (student log-prob of
the gold answer given a trace prefix). No student has yet been fine-tuned on compressed traces in this repo.

## Done
- Trace generation: instruct teacher k=5 (50 + extension to 100 problems), k=8 extension on 50, student pass@1..8, reasoning-teacher v2 (232 clean traces, 50 problems).
- Truncation diagnosis and fix for the reasoning teacher (3072 -> 8192 tokens, exact `hit_token_limit`).
- Measurement audit (Oct 14): relabelling, boundary-aware matching, gold-free cut -> `common/` + `analysis/analyze_traces_v3.py`.
- Valid comparison on the same 50 problems: reasoning-teacher traces are much more compressible (see RESULTS.md B-D).
- Glossary of project terms; results log with validity status.

## Left - integrity first (small, do before any new claim)
1. **Cut audit**: hand-check ~30 random cuts per teacher; report the share where the cut is the real conclusion (precision).
   Without this, "50% of words kept" has no error bar. -> script `diagnostics/audit_cuts.py` (to write).
2. **Token counts** (not words) for both teachers, full vs cut, to state a real cost saving.
3. **Rerun `score_by_difficulty.py` with v3 cuts** (prefix/middle/suffix-stripped/cut/full on one dataset, by tier). This is the
   experiment behind the old "adaptive beats fixed-fraction" claim; it is currently unsupported.
4. Instruct teacher on all **100 problems** with v3 (the 50-problem run above is the matched comparison; 100 gives power).
5. Reasoning teacher: more incorrect traces (only 28-31) - more problems and/or the 18 truncated rows regenerated at a 16k budget (Phoenix).

## Left - research
6. **Transfer test (the missing piece):** fine-tune the student on (a) full traces, (b) adaptive-cut traces, (c) random cut of equal length;
   evaluate held-out MATH accuracy. This is what turns the proxy into a result. Instruct-vs-reasoning teacher is the second axis.
7. Curriculum (Approach C): order traces by pass-rate difficulty; compare to random order. Bandit-style adaptive curriculum prototype after that.
8. Cross-family checks: Gemma now; Llama after licence acceptance.
9. Full-precision Phoenix reruns with matched k and 16k-32k token budgets (all current teacher results are 4/8-bit).
10. Meta-learning survey (carried over from earlier meetings).

## Open questions
- Is a "first statement of the final answer" the right commitment point for a backtracking reasoner? (Audit in item 1 will say how often it is an accident.)
- Does the signal-retention proxy predict actual fine-tuning gains? Only item 6 answers this.
