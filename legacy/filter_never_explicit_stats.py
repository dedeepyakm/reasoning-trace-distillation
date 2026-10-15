"""
Recompute adaptive-cut compression stats EXCLUDING never_explicit traces,
so the correct-vs-incorrect comparison isn't distorted by traces that
couldn't be cut at all (kept at 100% because no cut point exists).

Reads the adaptive_cut_traces.jsonl produced by adaptive_cut_analysis.py.

Usage:
    python3 filter_never_explicit_stats.py --cut-file adaptive_cut_traces.jsonl
"""

import argparse
import json


def main(cut_file):
    word_fracs = {"correct": [], "incorrect": []}
    excluded = {"correct": 0, "incorrect": 0}
    total = 0

    with open(cut_file) as f:
        for line in f:
            row = json.loads(line)
            total += 1
            label = "correct" if row["correct"] else "incorrect"
            if row.get("never_explicit"):
                excluded[label] += 1
                continue
            word_fracs[label].append(row["frac_words_kept"])

    print(f"Total traces: {total}")
    print(f"Excluded (never_explicit, kept at 100% by default): "
          f"{excluded['correct']} correct, {excluded['incorrect']} incorrect\n")

    print(f"{'Group':<12} {'N (cuttable)':>14} {'Avg % kept (words)':>20}")
    print("-" * 50)
    for label in ["correct", "incorrect"]:
        n = len(word_fracs[label])
        avg = 100 * sum(word_fracs[label]) / n if n else 0
        print(f"{label:<12} {n:>14} {avg:>19.1f}%")

    print(
        "\nThis isolates the real effect: among traces that DO state the "
        "answer explicitly somewhere before the end, how much earlier do "
        "correct traces commit to it compared to incorrect ones?"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cut-file", default="adaptive_cut_traces.jsonl")
    args = parser.parse_args()
    main(args.cut_file)
