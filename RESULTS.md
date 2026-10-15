# Results log

One entry per result. **Status**: VALID (current code in `common/`, checked against real examples) - SUPERSEDED (replaced by a later run) -
INVALID (produced by code with a known bug; do not quote) - NOT RE-AUDITED (earlier sessions; re-verify before quoting).

**Rule:** a number goes into a slide or report only if it has a VALID entry here.

All VALID numbers below come from the code at commit "Restructure repo..." plus the Oct-15 fixes to `common/` (multi-box answers,
`x=` prefixes, leading decimals, spacing). Commands that reproduce them are in `STATUS.md`.

## The Oct-14/15 audit (why earlier numbers are INVALID)

Opening real examples behind each statistic exposed these bugs, all now fixed in `common/answer_matching.py`:
1. Fallback answer extractor captured whole sentences ("...is 42 inches"); labelling used exact string equality
   ("90" vs `90^\circ`, "5" vs `x=5`, "10%" vs "10", `.35625` vs `0.35625`, `\text{Evelyn}` vs "Evelyn", "6 - 5i" vs "6-5i").
2. Several boxed answers in the final line ("\boxed{3}, \boxed{5}, \boxed{7}" vs gold "3, 5, 7") were compared by the last box only.
3. Plain-substring matching ("9" matched inside "196").
4. The cut point used the GOLD answer: it fired on problem restatements/lists, and was circular for incorrect traces
   (a wrong trace rarely states the gold, so "never explicit" partly meant "wrong"). Now the cut uses the model's OWN final answer,
   skips the opening sentences and problem restatements, questions ("is the answer 6?"), doubt ("Wait, that seems off") and numbers
   that also occur in the problem unless a strong cue (answer/therefore/final) is present.
5. Scorer ran out of GPU memory on very long traces; now computes logits for the answer tokens only.

Label changes were checked by eye with `diagnostics/show_relabelled.py`: reasoning 29 flips to correct (all correct), instruct 6 flips to correct
(all correct), 0 flips to incorrect.

---

## A. Reasoning-teacher generation (DeepSeek-R1-Distill-Qwen-7B, 50 MATH-500 problems, k=5, T=0.6, top_p=0.95, 4-bit)

| Run | Result | Status |
|---|---|---|
| v1 (max_new_tokens 3072) | 148/250 = 59.2% correct; 74 traces without `\boxed{}` (29.6%) - cause: truncation | SUPERSEDED |
| v2 (8192 tokens + fallback extractor) | 26/250 hit the token limit (10.4%); 18 of those have no extractable answer (7.2%) and were DROPPED -> 232 clean traces | VALID (generation facts) |
| v2 "74.8% accuracy", "73.7% early", "45.4% words kept", "9% retention" | from sentence-capturing extractor / substring matcher / gold-based cut | INVALID |

## B. Final comparison on the same 50 problems, k=5 (`analysis/analyze_traces_v3.py --with-difficulty`)

| | Reasoning teacher (n=232) | Instruct teacher, Qwen2.5-7B-Instruct (n=250) |
|---|---|---|
| Accuracy | 216/232 = 93.1% (Easy 97.6, Med 100, Hard 84.6) | 184/250 = 73.6% (Easy 96.0, Med 76.0, Hard 50.0) |
| Position of first statement of the model's own final answer: before last two sentences / last two / never | 181 / 22 / 29 | 35 / 139 / 76 |
| % correct within those buckets | 94.5 / 81.8 / 93.1 | 57.1 / 80.6 / 68.4 |
| Words kept after the cut: correct traces | 56.2% (N=189) | 92.5% (N=132) |
| Words kept after the cut: incorrect traces | 61.4% (N=14) | 87.7% (N=42) |

Caveats: (1) the reasoning accuracy excludes the 18 truncated traces with no answer; counting them as wrong gives 216/250 = 86.4%.
(2) Only 14 reasoning incorrect traces have a cut point, so the correct-vs-incorrect words-kept gap is not a finding.
(3) Instruct, Hard, answer stated early: 2 correct / 12 incorrect (n=14) - suggestive only.

## C. Cut precision audit (`diagnostics/audit_cuts.py`, reasoning teacher, 30 random cuts, labelled by hand-reading; Claude labelled, owner to confirm)

27 clean (the cut ends on the model stating its answer as a result), 2 hedged (`reasoning-46` samples 1 and 3: "might be 6" /
"does that mean n=6 ... Yes"), 1 early (`reasoning-23` sample 3: "16 - 11 is 5" is an intermediate that equals the final answer).
Precision 27/30 = 90% (95% interval 74-97%). Known remaining weakness: a coincidental intermediate equal to the final answer.
Instruct-teacher cuts have NOT been audited.

## D. Signal retention (`scoring/score_cut_v2.py`; student Qwen2.5-1.5B-Instruct; teacher-forced mean log-prob of the gold answer tokens)

Separation = mean logp(correct traces) - mean logp(incorrect traces). Retention = cut separation / full-trace separation.

| Teacher | Scored (correct / incorrect) | Cut separation | Full separation | Retention (95% bootstrap) |
|---|---|---|---|---|
| Reasoning | 60 / 16 | 1.476 | 1.611 | 91.6% (73-115%) |
| Instruct | 60 / 60 | 1.736 | 1.857 | 93.5% (87-99%) |

Reading: cutting at the model's own final answer keeps most of the measurable signal for both teachers. For the reasoning teacher that is with
about 44% of words removed; for the instruct teacher only about 8% were removed, so there was little to cut. The reasoning interval is wide
(16 incorrect traces) and includes both "no loss" and a ~27% loss. This is a PROXY: no student has been fine-tuned on cut traces yet.
All 76 / 120 pairs were scored; none skipped. Status: VALID.

## E. Earlier-session results (instruct teacher) - NOT RE-AUDITED
From project notes; scripts: `generation/extend_k_experiment*.py`, `generation/teacher_k8_extension.py`, `scoring/partition_heuristic.py`.
- k=2 conclusions reversed at k=5 (teacher confidence tracks difficulty, rho = -0.330; AUROC for own correctness rose). Pass-rate over k is the
  strongest difficulty signal (rho ~ -0.489). Student pass@k gains decelerate (k=8 practical cutoff). Student confidence separates correct/incorrect (AUROC 0.73).
- Partition heuristic: suffix >> middle > prefix at 50%/75%; stripping the last 2 sentences removes ~2/3 of the suffix advantage (mostly answer read-off). Gold-answer logp, not the cut logic; likely unaffected, verify.
- `scoring/score_by_difficulty.py` ("adaptive cut beats fixed-fraction heuristics, biggest margin on Hard"): used the gold-based cut -> **INVALID until rerun with the current cuts.**
- Old instruct findings from the gold-based cut ("~49% last-two", "never explicit => correct only ~8%"): INVALID.
