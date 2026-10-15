"""
Sanity-checks the adaptive cut on the reasoning teacher by showing, for a
few traces, exactly what text was KEPT (up to the first explicit answer
mention) vs what was CUT (everything after).

Why this matters: the scoring step showed adaptive_cut retaining only ~9%
of the full-trace signal for the reasoning teacher (vs ~71% for the
instruct teacher) -- a much bigger drop than expected. One plausible
mechanism: this model's long "wait, let me reconsider" backtracking style
means it often states the correct answer EARLY, then keeps re-deriving or
double-checking for a long time afterward. The adaptive cut stops at the
FIRST mention, so it may be throwing away a later, more conclusive
confirmation that the student model actually relies on -- unlike the
instruct teacher, whose traces are shorter and whose first mention is
usually close to its real conclusion anyway.

This script doesn't prove or disprove that -- it just lets you look
directly at a few examples and judge for yourself, the same way
inspect_no_boxed.py let us directly see the truncation issue instead of
guessing from statistics alone.

Usage:
    python inspect_adaptive_cut_reasoning.py --cut-file adaptive_cut_traces_reasoning.jsonl --n 4
"""

import argparse
import json


def main(cut_file, n, cut_tail_chars):
    shown = 0
    with open(cut_file) as f:
        for line in f:
            if shown >= n:
                break
            row = json.loads(line)
            kept_text = row.get("adaptive_truncated_text", "")
            frac_kept = row.get("frac_words_kept")
            never_explicit = row.get("never_explicit")
            if never_explicit:
                continue  # not interesting here -- whole trace kept by convention

            shown += 1
            print("=" * 78)
            print(f"id={row['id']}  correct={row.get('correct')}  "
                  f"frac_words_kept={frac_kept}  total_sentences={row.get('total_sentences')}  "
                  f"cut_sentence_idx={row.get('cut_sentence_idx')}")
            print("-" * 78)
            print("LAST ~300 CHARS OF WHAT WAS KEPT (i.e. the sentence the cut stops at):")
            print(kept_text[-300:])
            print()

    if shown == 0:
        print("No non-never-explicit rows found (or file is empty/all never-explicit).")
    else:
        print(f"\nShown {shown} example(s). Judge for yourself: does the kept text end on a "
              "confident, conclusive statement of the answer, or does it read like a guess "
              "the model was still going to second-guess? (We can't see the cut-off part "
              "directly from this file -- it's not saved -- but a low frac_words_kept with "
              "an early, tentative-sounding cut sentence would support the backtracking "
              "hypothesis; a late, confident one would argue the low retention score has a "
              "different cause, e.g. the student model struggling with this model's prompt "
              "style generally.)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cut-file", required=True)
    parser.add_argument("--n", type=int, default=4)
    parser.add_argument("--cut-tail-chars", type=int, default=300)
    args = parser.parse_args()
    main(args.cut_file, args.n, args.cut_tail_chars)
