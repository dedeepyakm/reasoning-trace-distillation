# Error log

One row per error. Add a row when something breaks, fill in the cause and fix once known, and set status to `fixed` or `open`.
Failed runs from `run_step` in the Colab notebook are added automatically (cause and fix are then filled in by hand).

| When | Where | Error | Cause | Fix | Status |
|---|---|---|---|---|---|
| Oct 2026 | answer extraction | Answers wrongly marked incorrect | Extractor captured whole sentences; substring match ("9" inside "196") | Boundary- and cue-aware matcher in `common/answer_matching.py` | fixed |
| Oct 2026 | cut logic | "never explicit" partly meant "wrong" | Old cut used the gold answer (circular) | Gold-free cut: first sentence where the model states its own answer | fixed |
| Oct 2026 | cut logic | Cuts landed on restated problem text and intro lists | No check for restatement, questions, doubt, or numbers copied from the problem | `is_restatement`, doubt and problem-number checks | fixed |
| Oct 2026 | label matching | `x=5` vs `5` marked incorrect | Leading `x=` not stripped | Strip a leading single-letter `x=` | fixed |
| Oct 2026 | label matching | Correct answers with several `\boxed{}` marked incorrect (gold "3, 5, 7") | Only the last boxed value compared | `multi_boxed_match` | fixed |
| Oct 2026 | label matching | `6 - 5i` flipped from correct to incorrect | Spaced vs space-free comparison | Fallback tries spaced then space-free | fixed |
| Oct 2026 | label matching | `.35625` vs `0.35625` (5 instruct traces wrongly flipped) | Leading decimal not normalised | `strip_wrappers` adds the leading 0 | fixed |
| Oct 2026 | reasoning generation | 29.6% of v1 traces had no `\boxed{}` | 3072-token limit truncated the traces | 8192 tokens; exact `hit_token_limit` flag | fixed |
| Oct 2026 | reasoning generation | 18 traces still truncated at 8192 with no answer (8 problems, problem 43 fully lost) | Reasoning traces longer than the limit | Dropped for now; regenerate at 16k (open item) | open |
| Oct 2026 | scoring | CUDA out of memory on long reasoning traces (local RTX 5070, 8 GB) | Full logits for long sequences; root cause not confirmed | `logits_to_keep`; OOM pairs skipped and reported; 0 skipped in final runs | fixed, cause unconfirmed |
| Oct 2026 | scoring | `expandable_segments` setting did nothing | Not supported on Windows | Ignore; use Colab GPU for long traces | closed |
| Oct 2026 | comparison | "Same 50 problems" claim in RESULTS.md was wrong | Teachers ran on different problems; only 5 overlap | Wording corrected; like-for-like run planned | fixed (wording), open (run) |
