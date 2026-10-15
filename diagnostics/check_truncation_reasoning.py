"""
Diagnostic: did generation hit max_new_tokens before reaching \\boxed{} on the
reasoning-teacher traces?

Why this matters: traces_reasoning_teacher_k5.jsonl was generated with
max_new_tokens=3072. Several problems show every sample at 2000+ words
(~2600-2900+ tokens) and scored incorrect. If generation hit the token cap
before the model wrote \\boxed{...}, that sample has predicted_answer=None
and gets scored "incorrect" even though we never actually saw a final
answer -- a budget artifact, not a reasoning failure. Running the position/
compression analysis on unflagged data like this would dump these into the
"never_explicit" bucket for the wrong reason (truncation, not a model that
reasons without ever stating the answer), muddying the exact comparison
we're trying to make against the instruct teacher.

This script does NOT re-run the model. It just flags which existing rows
are suspect, using word count as a proxy for closeness to the token budget
(no exact token count was logged during generation).

Usage:
    python check_truncation_reasoning.py --trace-file traces_reasoning_teacher_k5.jsonl --max-new-tokens 3072
"""

import argparse
import json
from collections import defaultdict

# Rough, conservative proxy: assume ~1.2 words per token for this kind of
# LaTeX-heavy math CoT (slightly lower than plain English ~1.3, since LaTeX
# commands and symbols often count as more tokens per "word"). A trace
# within 85% of the token budget by this proxy is flagged as suspect.
WORDS_PER_TOKEN = 1.2
SUSPECT_FRACTION = 0.85


def main(trace_file, max_new_tokens):
    threshold_words = max_new_tokens * WORDS_PER_TOKEN * SUSPECT_FRACTION

    total = 0
    no_boxed = 0
    no_boxed_and_long = 0
    by_problem = defaultdict(list)

    with open(trace_file) as f:
        for line in f:
            row = json.loads(line)
            total += 1
            wc = len(row.get("reasoning_text", "").split())
            has_boxed = row.get("predicted_answer") is not None
            by_problem[row["id"]].append((wc, has_boxed, row.get("correct")))
            if not has_boxed:
                no_boxed += 1
                if wc >= threshold_words:
                    no_boxed_and_long += 1

    print(f"Total traces: {total}")
    print(f"No \\boxed{{}} found at all: {no_boxed} ({100*no_boxed/total:.1f}%)")
    print(
        f"No \\boxed{{}} AND word count >= {threshold_words:.0f} "
        f"(suspect truncation): {no_boxed_and_long} ({100*no_boxed_and_long/total:.1f}%)"
    )

    print(f"\n{'='*78}\nPROBLEMS WHERE EVERY SAMPLE IS SUSPECT (likely needs regeneration)\n{'='*78}")
    fully_suspect_problems = []
    for pid, samples in by_problem.items():
        if all((not has_boxed and wc >= threshold_words) for wc, has_boxed, _ in samples):
            fully_suspect_problems.append(pid)
            lens = [wc for wc, _, _ in samples]
            print(f"  {pid}: all {len(samples)} samples suspect, word counts {lens}")

    print(f"\n{len(fully_suspect_problems)} / {len(by_problem)} problems are fully suspect.")

    print(f"\n{'='*78}\nACCURACY WITH vs WITHOUT SUSPECT ROWS\n{'='*78}")
    all_correct = sum(1 for s in by_problem.values() for wc, hb, c in s if c)
    all_n = sum(len(s) for s in by_problem.values())
    clean_correct = sum(
        1 for s in by_problem.values() for wc, hb, c in s
        if c and not (not hb and wc >= threshold_words)
    )
    clean_n = sum(
        1 for s in by_problem.values() for wc, hb, c in s
        if not (not hb and wc >= threshold_words)
    )
    print(f"All traces:            {all_correct}/{all_n} = {100*all_correct/all_n:.1f}%")
    if clean_n > 0:
        print(f"Excluding suspect rows: {clean_correct}/{clean_n} = {100*clean_correct/clean_n:.1f}%")

    if fully_suspect_problems:
        print(
            "\nRecommendation: regenerate these problem indices with a higher "
            "--max-new-tokens (try 6144) before running the position/compression "
            "analysis, so the never_explicit bucket reflects real model behavior, "
            "not a token-budget cutoff. Use --start-index / a small custom problem "
            "list, or just delete these rows from the output file and rerun "
            "generation for the same --out path with a higher budget (the script "
            "resumes by skipping ids already present, so you'd need to remove "
            "their rows first)."
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--trace-file", required=True)
    parser.add_argument("--max-new-tokens", type=int, default=3072)
    args = parser.parse_args()
    main(args.trace_file, args.max_new_tokens)
