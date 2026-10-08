"""
Shows the tail of a few traces where no \\boxed{} answer was found, so we can
see directly whether generation was cut off mid-sentence (truncation -> fix
by raising --max-new-tokens and regenerating) or ended cleanly in some other
format (an extraction bug -> fix in code, no regeneration needed).

No GPU needed -- just reads the existing trace file.

Usage:
    python inspect_no_boxed.py --trace-file traces_reasoning_teacher_k5.jsonl --n 6 --tail-chars 500
"""

import argparse
import json


def main(trace_file, n, tail_chars):
    shown = 0
    with open(trace_file) as f:
        for line in f:
            if shown >= n:
                break
            row = json.loads(line)
            if row.get("predicted_answer") is not None:
                continue
            shown += 1
            text = row.get("reasoning_text", "")
            print("=" * 78)
            print(f"id={row['id']}  sample_idx={row.get('sample_idx')}  "
                  f"word_count={len(text.split())}  gold_answer={row.get('answer')}")
            print("-" * 78)
            print("...END OF TRACE:")
            print(text[-tail_chars:])
            print()

    if shown == 0:
        print("No rows with predicted_answer=None found (or trace file is empty).")
    else:
        print(f"\nShown {shown} example(s). Look for: does the text stop mid-word/mid-equation "
              "(truncation) or end on a complete sentence/thought with the answer stated in some "
              "other form (extraction format mismatch)?")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--trace-file", required=True)
    parser.add_argument("--n", type=int, default=6, help="how many examples to show")
    parser.add_argument("--tail-chars", type=int, default=500, help="how many trailing characters to print per example")
    args = parser.parse_args()
    main(args.trace_file, args.n, args.tail_chars)
