"""
Removes genuinely-truncated rows from the v2 reasoning-teacher traces before
running the main position/compression/scoring analysis.

Why this is needed: teacher_trace_generation_reasoning_v2.py raised the
token budget to 8192 and fixed the 29.6% no-answer rate from v1 down to a
10.4% hit_token_limit rate -- but not all of those 26 rows are bad. Looking
at BOTH flags together:

  - hit_token_limit=True AND predicted_answer is not None
        -> the model's answer was still extracted (boxed or fallback) right
           around the token boundary. This is a long trace, not missing
           information. KEEP.
  - hit_token_limit=True AND predicted_answer is None
        -> generation used the entire budget and STILL never produced an
           extractable answer. This is real information loss (the same
           failure mode v1 had, just rarer). These rows would otherwise
           land in the "never_explicit" bucket for the wrong reason
           (truncation, not genuine model behavior) when run through
           answer_position_analysis.py / difficulty_stratified_analysis.py.
           DROP.

This keeps the same spirit as filter_never_explicit_stats.py: don't silently
let a confound (here: truncation) masquerade as a real finding (here: "the
model often never states its answer").

Usage:
    python filter_truncated_reasoning.py --in traces_reasoning_teacher_k5_v2.jsonl --out traces_reasoning_teacher_k5_v2_clean.jsonl
"""

import argparse
import json


def main(in_path, out_path):
    kept = 0
    dropped = 0
    dropped_ids = []

    with open(in_path) as f_in, open(out_path, "w") as f_out:
        for line in f_in:
            row = json.loads(line)
            genuinely_truncated = (
                row.get("hit_token_limit") is True and row.get("predicted_answer") is None
            )
            if genuinely_truncated:
                dropped += 1
                dropped_ids.append(f"{row['id']}/sample{row.get('sample_idx')}")
                continue
            f_out.write(json.dumps(row) + "\n")
            kept += 1

    total = kept + dropped
    print(f"Read {total} traces from {in_path}")
    print(f"Kept:    {kept} ({100*kept/total:.1f}%)")
    print(f"Dropped: {dropped} ({100*dropped/total:.1f}%) -- hit token limit with no extractable answer")
    if dropped_ids:
        print("\nDropped rows (genuinely truncated, no information):")
        for d in dropped_ids:
            print(f"  {d}")
    print(f"\nClean file written to {out_path}")
    print(
        "\nNext: run the existing pipeline scripts on this clean file, e.g.\n"
        f"  python answer_position_analysis.py --trace-file {out_path}\n"
        f"  python adaptive_cut_analysis.py --trace-file {out_path} --out adaptive_cut_traces_reasoning.jsonl"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--in", dest="in_path", required=True)
    parser.add_argument("--out", default="traces_reasoning_teacher_k5_v2_clean.jsonl")
    args = parser.parse_args()
    main(args.in_path, args.out)
