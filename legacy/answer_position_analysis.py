"""
Answer-position analysis -- no GPU/model needed, pure text analysis.

For each trace in a trace file, finds the FIRST sentence where the gold answer
is explicitly stated (as text), then buckets each trace into one of:
  - never_explicit   : answer never appears explicitly in the reasoning text
  - early_or_middle  : first explicit mention is before the last 2 sentences
  - last_two         : first explicit mention is within the last 2 sentences

Then reports, per bucket: how many traces were correct vs incorrect. This
directly answers "when the answer is stated early/mid vs only at the very
end, how often was that trace actually right?"

Usage:
    python3 answer_position_analysis.py --trace-file traces_teacher_k5.jsonl
"""

import argparse
import json
import re
from collections import defaultdict


def normalize_for_search(text):
    """Loose normalization for substring matching -- lowercase, strip $ and
    extra whitespace. Not a full math-equivalence checker; a reasonable
    heuristic for tonight, not a rigorous symbolic match."""
    text = text.lower().strip()
    text = text.replace("$", "").replace(",", "")
    text = re.sub(r"\s+", " ", text)
    return text


def split_sentences(text):
    parts = re.split(r'(?<=[.?!])\s+', text.strip())
    return [p for p in parts if p]


def find_first_mention_sentence(sentences, gold_answer):
    """Returns the index of the first sentence containing the gold answer as
    a normalized substring, or None if it never appears."""
    norm_gold = normalize_for_search(gold_answer)
    if not norm_gold:
        return None
    for i, sent in enumerate(sentences):
        if norm_gold in normalize_for_search(sent):
            return i
    return None


def bucket_trace(reasoning_text, gold_answer):
    sentences = split_sentences(reasoning_text)
    if not sentences:
        return "never_explicit"
    idx = find_first_mention_sentence(sentences, gold_answer)
    if idx is None:
        return "never_explicit"
    if idx >= len(sentences) - 2:
        return "last_two"
    return "early_or_middle"


def main(trace_file):
    counts = defaultdict(lambda: {"correct": 0, "incorrect": 0})
    total_rows = 0

    with open(trace_file) as f:
        for line in f:
            row = json.loads(line)
            reasoning = row.get("reasoning_text")
            gold = row.get("answer")
            correct = row.get("correct")
            if reasoning is None or gold is None or correct is None:
                continue
            bucket = bucket_trace(reasoning, gold)
            counts[bucket]["correct" if correct else "incorrect"] += 1
            total_rows += 1

    print(f"Analyzed {total_rows} traces from {trace_file}\n")
    print(f"{'Bucket':<20} {'Correct':>8} {'Incorrect':>10} {'Total':>7} {'% Correct':>10}")
    print("-" * 60)
    for bucket in ["early_or_middle", "last_two", "never_explicit"]:
        c = counts[bucket]["correct"]
        i = counts[bucket]["incorrect"]
        total = c + i
        pct = (100 * c / total) if total > 0 else 0.0
        label = {
            "early_or_middle": "Early/intermediate",
            "last_two": "Last 2 sentences",
            "never_explicit": "Never explicit",
        }[bucket]
        print(f"{label:<20} {c:>8} {i:>10} {total:>7} {pct:>9.1f}%")

    print("\nInterpretation:")
    print("If 'Last 2 sentences' has a much higher % correct than 'Early/intermediate',")
    print("that suggests the model tends to only explicitly commit to a stated answer")
    print("near the end when it has actually reasoned its way to the right one --")
    print("i.e. stating the answer late is itself a signal of confidence/correctness,")
    print("not just a mechanical restatement.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--trace-file", required=True)
    args = parser.parse_args()
    main(args.trace_file)
