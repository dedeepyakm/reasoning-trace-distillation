# Glossary — project-specific terms

These terms are **not** standard ML terminology — they're definitions we
created for this project, implemented in specific functions in the analysis
scripts. This document exists so that anyone reading our results (including
us, months from now) knows exactly what each term means operationally, not
just roughly. Every definition below points at the exact function/logic that
implements it.

---

## The basic unit: a "trace"

A **trace** is one full chain-of-reasoning generation from the teacher model
for one MATH-500 problem. Because we sample **k=5** times per problem (i.e.
run the teacher 5 separate times on the same problem, with sampling/
temperature so the 5 runs differ), **one problem produces 5 traces**. So "500
traces" means 100 problems × 5 samples each — not 500 distinct problems.

Each trace is stored as one JSON row with (at minimum):
- `id` — which problem it's for (e.g. `math500-reasoning-17` = problem index 17)
- `reasoning_text` — the full text the model generated (its reasoning, in
  prose/LaTeX, from start to wherever it stopped)
- `answer` — the gold (correct) answer for that problem, taken from the
  MATH-500 dataset itself
- `correct` — whether *this specific trace's* final extracted answer matched
  the gold answer (see "correct / incorrect" below)

**Important consequence:** every statistic in our reports (accuracy,
bucket percentages, compression ratios) is computed **per trace**, not per
problem. "83.2% accuracy on Easy" means 83.2% of the individual sampled
traces for Easy-tier problems were correct — it is not "83.2% of Easy
problems were solved at least once in 5 tries" (that would be a different,
higher number, since a problem only needs one correct trace out of 5 to
count as "solvable").

---

## "Correct" / "Incorrect"

This is a **per-trace** label, decided independently for every one of the
k=5 samples, by comparing the model's own final stated answer against the
gold answer:

1. We find the model's final answer by taking the **last** `\boxed{...}` in
   the trace (brace-depth-aware, so `\boxed{\frac{1}{2}}` isn't cut short at
   the first `}`). If a model never used `\boxed{}` at all, we now (as of
   today's v2 reasoning-teacher script) also try a handful of fallback
   patterns like `**Answer:** ...` or `the answer is ...`.
2. Both the extracted answer and the gold answer are run through
   `normalize_answer()` — this strips spaces, LaTeX spacing commands
   (`\!`, `\,`), a trailing period, `\left( ... \right)` wrappers, and
   collapses `\dfrac`/`\tfrac` to `\frac`, so that trivially-equivalent
   LaTeX (e.g. `\dfrac{1}{2}` vs `\frac{1}{2}`) doesn't get marked wrong for
   a formatting difference.
3. `correct = (normalized extracted answer == normalized gold answer)`.

If no answer could be extracted at all (no `\boxed{}` and no fallback match),
the trace is automatically `correct = False` — **this is why a truncated,
cut-off trace shows up as "incorrect" even though the model may never have
actually gotten the chance to answer.** (This is exactly the issue we caught
today with the reasoning-teacher traces — see the project README's "Known
limitations" section.)

---

## "Where the answer is first explicitly stated" — the position buckets

This is the heart of `answer_position_analysis.py` and
`difficulty_stratified_analysis.py`. The question we're asking is:
**at what point in its own reasoning does the model first write down, in
prose, the same value as the gold answer** — regardless of whether that
mention is wrapped in `\boxed{}` or just appears in a sentence of reasoning?

### Step 1 — split the trace into sentences

`split_sentences()` breaks `reasoning_text` into a list of sentences using
`.`/`?`/`!` as boundaries (a regex sentence split, not perfect for every
edge case in LaTeX, but consistent across all traces).

### Step 2 — normalize for searching

`normalize_for_search()` lowercases the text, strips `$` and `,`, and
collapses whitespace — this is a **separate, looser** normalization than the
`normalize_answer()` used for correctness-checking above. It exists only to
make substring search more forgiving (so `"x = 5"` and `"x=5"` are treated
the same when searching for a mention), not to judge mathematical
correctness.

### Step 3 — does this sentence "contain" the gold answer?

`sentence_contains_answer()` checks whether the normalized gold answer
appears as a substring of the normalized sentence — **with one important
exception**: if the gold answer is a bare number (e.g. `"2"`), we require a
non-digit character (or string boundary) on both sides, so the sentence
*"...the answer is 12 because..."* does **not** count as mentioning gold
answer `"2"`. This was a real bug we found and fixed (an earlier version of
this script used plain substring matching, which meant `"2"` falsely
matched inside `"12"`, `"20"`, `"2.5"`, etc., inflating the "early/
intermediate" bucket with false positives). Non-numeric gold answers
(fractions, expressions, words) still use plain substring matching, since
word-boundary rules don't apply cleanly to LaTeX expressions.

### Step 4 — find the FIRST sentence that mentions it

`find_first_mention_sentence()` walks the sentence list **in order** and
returns the index of the **first** sentence that mentions the gold answer.
This is deliberate: a trace might state the right answer early as a guess,
work through more reasoning, and then restate it at the end — we only care
about the *first* time it appears, because that's the earliest point at
which the compressed/truncated version of the trace could plausibly stop
without losing the answer.

### Step 5 — assign the bucket

Given `idx` = the index of that first mention (or `None` if never found)
and `len(sentences)` = total sentence count in the trace:

| Bucket | Rule (exact code logic) | Plain-language meaning |
|---|---|---|
| **`never_explicit`** | `idx is None` (no sentence ever mentions the gold answer) | The model's reasoning prose never states the value that happens to be the correct answer, anywhere in the trace — even if it eventually outputs that value inside `\boxed{}` at the very end without ever "saying" it in a sentence first, or if it answers wrong and the correct value genuinely never comes up. |
| **`last_two`** | `idx >= len(sentences) - 2` | The *first* mention of the gold answer falls within the final two sentences of the trace — i.e., the model doesn't commit to/state this value until it's concluding. This is the "answer is near the end" pattern we originally assumed was universal. |
| **`early_or_middle`** | `idx < len(sentences) - 2` (and not None) | The model states the gold-answer value somewhere **before** the final two sentences — meaning it arrived at (or guessed) the right value earlier, then kept reasoning/verifying afterward. This is the pattern the supervisor specifically asked us to check for (Sept 24 feedback): "the answer may already be stated earlier, mid-trace, before the formal concluding restatement." |

**A subtlety worth remembering:** `never_explicit` does NOT mean "the model
got no answer" or "the trace was truncated." It specifically means *the
literal gold-answer value never appears as a textual match in any sentence*
— this can happen for a perfectly complete, well-formed trace where the
model's language just never phrases things in a way our substring search
catches (e.g. it says "which simplifies to one half" instead of writing
"1/2"), or for a trace that's simply wrong and never comes close to the
right value, or (the failure mode we found today) for a trace that got cut
off by the token budget before it ever stated anything resembling a final
answer.

---

## "Adaptive cut" / "frac_words_kept"

Once we know the bucket and the index of the first explicit mention, the
**adaptive cut** (`adaptive_cut_analysis.py` / the `_by_difficulty` version)
keeps everything in the trace **up to and including that first
answer-mention sentence**, and discards everything after it (verification,
restatement, double-checking). This is "adaptive" because the cut point is
different for every single trace — it's not a fixed 50%/75% fraction like
the other baselines.

- `frac_words_kept` = (word count of the kept text) / (word count of the
  original full trace).
- For `never_explicit` traces, there's no valid cut point (nowhere to
  cut), so by convention the **entire original trace is kept**
  (`frac_words_kept = 1.0`) — this is why "never-explicit" traces must be
  excluded when computing an honest average compression ratio
  (`filter_never_explicit_stats.py` exists specifically to do this
  exclusion), otherwise they silently drag the average "% kept" up toward
  100% regardless of how well adaptive cutting works on the traces where
  it actually had something to cut.

---

## "Signal retention" / "log-prob separation" scoring

This measures whether a **compressed** trace (adaptive-cut, or a fixed
prefix/middle/suffix fraction) still gives the **student model** enough
information to be confident in the right answer — i.e., did we throw away
reasoning that the student actually needed, or just redundant restatement?

1. Build a prompt: the original problem + "Partial reasoning so far:
   `<the truncated/compressed trace>`" + `Answer: \boxed{`
2. Feed this to the **student** model (not the teacher) and compute the
   average log-probability it assigns to the tokens of the **gold answer**
   (teacher-forced — the student doesn't generate, we just score how
   likely it considers the correct continuation).
3. Do this separately for traces where the *teacher* was correct and where
   the *teacher* was incorrect, for each truncation strategy.
4. **Separation** = (mean log-prob on correct-labeled traces) − (mean
   log-prob on incorrect-labeled traces). A bigger separation means the
   compressed trace is doing a better job of making the student confident
   specifically when it should be (and not when the underlying reasoning
   was actually wrong) — i.e., the compression preserved the *useful*
   signal, not just made the student agree with everything.

---

## Difficulty tiers (Easy / Medium / Hard)

MATH-500 ships a `level` field, 1 (easiest) to 5 (hardest), per problem.
We group these into three tiers (`TIER_NAMES` in
`difficulty_stratified_analysis.py`):

| Tier | MATH-500 levels |
|---|---|
| Easy | 1, 2 |
| Medium | 3 |
| Hard | 4, 5 |

All position-bucket, compression, and scoring statistics can be (and in the
"this week" results, are) broken out by tier, to test whether a pattern
found on the full dataset is being driven mostly by the easy problems or
genuinely holds on hard ones too.
