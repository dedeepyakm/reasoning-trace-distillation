# Results log

One entry per result. **Status** is one of:
- **VALID** - produced by current code (shared matcher in `common/`), checked against examples.
- **SUPERSEDED** - replaced by a later run; kept for history.
- **INVALID** - produced by code with a known bug; do not quote.
- **NOT RE-AUDITED** - from earlier sessions; the code path was not part of the Oct-14 audit. Re-verify before quoting.

Rule: a number goes into a slide or report only if it has a VALID entry here.

## The Oct-14 audit (why several entries below are INVALID)

Opening real examples showed three measurement bugs:
1. **Answer extraction/labelling.** The v2 fallback extractor captured whole sentences ("...is 42 inches") and
   labelling used exact string equality ("90" vs `90^\circ`), so correct answers were labelled incorrect.
2. **Substring matching** in `answer_position_analysis.py` / `adaptive_cut_analysis.py` (gold "9" matched inside "196").
3. **Cut point.** "First sentence containing the GOLD answer" (a) fires on problem restatements and intro lists,
   (b) is circular for incorrect traces: a wrong trace rarely states the gold, so it lands in "never explicit"
   and "never explicit => mostly wrong" is partly just "wrong => gold absent".
Fix: `common/answer_matching.py` + `analysis/analyze_traces_v3.py` (cut at the model's OWN final answer, gold-free).

---

## A. Reasoning-teacher generation (DeepSeek-R1-Distill-Qwen-7B, 50 problems, k=5, T=0.6, 4-bit)

| Run | Result | Status |
|---|---|---|
| v1 (max_new_tokens=3072) | 148/250 = 59.2% correct, 74 no-`\boxed{}` (29.6%) | SUPERSEDED - cause was truncation |
| v2 (8192 tokens + fallback extractor) | 26/250 hit token limit (10.4%); 18 of those have no answer (7.2%) | VALID (generation facts) |
| v2 accuracy "74.8%" | from the sentence-capturing extractor | INVALID - see next row |
| v2 clean set (18 dropped) relabelled by `analyze_traces_v3.py` | 232 traces; 15 incorrect->correct, 1 correct->incorrect; accuracy 201/232 = 86.6% (Easy 91.6, Medium 100, Hard 73.6) | VALID |

## B. Reasoning teacher: answer position and compression (gold-free cut, n=232)

| Quantity | Value | Status |
|---|---|---|
| First statement of final answer: early/middle / last-two / never | 181 / 22 / 29 | VALID |
| % correct in those buckets | 86.7 / 81.8 / 89.7 | VALID |
| Words kept after cut: correct (N=175) / incorrect (N=28) | 50.6% / 45.8% | VALID, upper-bound-ish (see caveat) |
| Old numbers: 73.7% early, 45.4% kept, never-explicit 34% correct | gold-based cut | INVALID |

Caveat: the cut still fires on a coincidental intermediate that equals the final answer (seen in 2 of 8 inspected
traces: `reasoning-9`, `reasoning-4`). Precision of the cut is not yet measured (see STATUS: cut audit).

## C. Instruct teacher (Qwen2.5-7B-Instruct, same 50 problems, k=5), gold-free cut, n=250

| Quantity | Value | Status |
|---|---|---|
| Accuracy after relabelling | 179/250 = 71.6% (Easy 91.0, Medium 76.0, Hard 50.0); 6 up / 5 down vs stored labels | VALID |
| Position: early/middle / last-two / never | 39 / 140 / 71 | VALID |
| Words kept: correct (N=131) / incorrect (N=48) | 92.4% / 83.1% | VALID |
| Hard + early answer: 2 correct / 16 incorrect | n=18, suggestive only | VALID, small n |
| Earlier "~49% last-two, ~8% correct when never explicit" etc. | gold-based cut | INVALID |

## D. Signal retention (`scoring/score_cut_v2.py`): student Qwen2.5-1.5B log-prob of gold answer

Separation = mean logp(correct traces) - mean logp(incorrect traces). Retention = cut separation / full separation.

| Teacher | n scored (correct/incorrect) | Cut sep. | Full sep. | Retention (95% bootstrap) | Status |
|---|---|---|---|---|---|
| Reasoning | 60 / 31 | 1.115 | 0.994 | 112% (81-159%) | VALID, wide interval: "no detectable loss", not "better" |
| Instruct | 60 / 60 | 1.644 | 1.858 | 88.5% (81-95%) | VALID, but only ~8% of words were cut |
| Reasoning (old scorer, 40/6) "78%" and "9%" | | | | | INVALID / too small |
| Instruct "~71%" (old cut logic) | | | | | INVALID |

Reading: the reasoning teacher's traces are far more compressible (answer reached ~halfway; about half the words
removed with no detectable loss); the instruct teacher's traces barely have anything to cut.

## E. Earlier-session results (instruct teacher) - NOT RE-AUDITED

Taken from project notes, not rerun in this audit. Scripts involved: `generation/extend_k_experiment*.py`,
`generation/teacher_k8_extension.py`, `scoring/partition_heuristic.py`.
- Teacher k=2 conclusions were reversed at k=5 (confidence tracks difficulty, rho = -0.330; AUROC for own correctness rose).
- Pass-rate over k samples is the strongest difficulty signal (rho ~ -0.489).
- Student pass@k shows decelerating gains, supporting k=8 as a practical cutoff.
- Student confidence separates correct from incorrect (AUROC 0.73).
- Partition heuristic: suffix >> middle > prefix at 50%/75%; stripping the last 2 sentences removes ~2/3 of the
  suffix advantage (mostly answer read-off). Uses gold-answer logp, not the cut logic, so likely unaffected, but verify.
- `scoring/score_by_difficulty.py` ("adaptive cut beats every fixed-fraction heuristic, largest margin on Hard"):
  used the gold-based cut -> **INVALID until rerun with v3 cuts**. The README claim built on it has been removed.
