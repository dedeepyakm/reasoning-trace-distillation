# legacy/ - superseded scripts, kept for history

Do not use these for new results. Their outputs are marked INVALID or SUPERSEDED in `../RESULTS.md`.

| Script | Why superseded |
|---|---|
| `answer_position_analysis.py`, `adaptive_cut_analysis.py`, `filter_never_explicit_stats.py` | Plain substring matching ("9" matches in "196") and a gold-answer cut. Replaced by `analysis/analyze_traces_v3.py`. |
| `difficulty_stratified_analysis.py` | Fixed digit boundaries but still cuts on the GOLD answer and still counts problem restatements. Replaced by `analysis/analyze_traces_v3.py --with-difficulty`. |
| `score_adaptive_cut.py` | One trace per problem, tiny incorrect group, gold-based cuts. Replaced by `scoring/score_cut_v2.py`. |
| `teacher_trace_generation_reasoning.py` | v1: 3072-token budget truncated 29.6% of traces. Replaced by `generation/teacher_trace_generation_reasoning_v2.py`. |
| `analyze_traces_fixed.py` | Intermediate fix (correct relabelling + boundary/cue matching, still gold-based cut). |
| `inspect_adaptive_cut_reasoning.py` | One-off inspection; `diagnostics/inspect_cut_diverse.py` replaced it. |
