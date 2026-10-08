"""
Difficulty-stratified answer-position + adaptive-cut analysis.

Two things this adds on top of last week's answer_position_analysis.py /
adaptive_cut_analysis.py:

1. BUG FIX: the original "does this sentence contain the gold answer" check
   used plain substring containment. For a numeric gold answer like "2", that
   also matches inside "12", "20", "2.5", etc -- a false positive that could
   inflate the "early/intermediate" bucket. This version requires a
   non-digit boundary around a purely numeric answer before counting it as a
   match. Non-numeric answers (fractions, expressions) still use substring
   containment, since word-boundary rules don't cleanly apply to LaTeX.

2. DIFFICULTY BREAKDOWN: MATH-500 ships a "level" field (1-5, 5 = hardest).
   Every table below is split into Easy (level 1-2), Medium (level 3), and
   Hard (level 4-5), so we can show the pipeline holds on hard problems
   specifically, not just in an average dominated by easy ones.

Also writes adaptive_cut_traces_by_difficulty.jsonl (same schema as last
week's adaptive_cut_traces.jsonl, plus a "difficulty_tier" field) for the
follow-up GPU scoring script, score_by_difficulty.py.

Usage:
    python3 difficulty_stratified_analysis.py --trace-file traces_teacher_k5_combined.jsonl --out adaptive_cut_traces_by_difficulty.jsonl
"""

import argparse
import json
import re
from collections import defaultdict

from datasets import load_dataset

TIER_NAMES = {1: "Easy", 2: "Easy", 3: "Medium", 4: "Hard", 5: "Hard"}
TIER_ORDER = ["Easy", "Medium", "Hard"]


def normalize_for_search(text):
    text = text.lower().strip()
    text = text.replace("$", "").replace(",", "")
    text = re.sub(r"\s+", " ", text)
    return text


def split_sentences(text):
    parts = re.split(r'(?<=[.?!])\s+', text.strip())
    return [p for p in parts if p]


def is_purely_numeric(norm_gold):
    return re.match(r'^-?\d+(\.\d+)?$', norm_gold.replace(" ", "")) is not None


def sentence_contains_answer(norm_sent, norm_gold):
    """Fixed matching: numeric answers need a non-digit boundary on both
    sides, so gold "2" does not match inside "12" or "20". Non-numeric
    answers (fractions, expressions, words) fall back to substring
    containment, same as before."""
    if not norm_gold:
        return False
    if is_purely_numeric(norm_gold):
        pattern = r'(?<!\d)' + re.escape(norm_gold) + r'(?!\d)'
        return re.search(pattern, norm_sent) is not None
    return norm_gold in norm_sent


def find_first_mention_sentence(sentences, gold_answer):
    norm_gold = normalize_for_search(gold_answer)
    if not norm_gold:
        return None
    for i, sent in enumerate(sentences):
        if sentence_contains_answer(normalize_for_search(sent), norm_gold):
            return i
    return None


def bucket_trace(reasoning_text, gold_answer):
    sentences = split_sentences(reasoning_text)
    if not sentences:
        return "never_explicit", None, 0
    idx = find_first_mention_sentence(sentences, gold_answer)
    if idx is None:
        return "never_explicit", None, len(sentences)
    if idx >= len(sentences) - 2:
        return "last_two", idx, len(sentences)
    return "early_or_middle", idx, len(sentences)


def load_level_lookup():
    dataset = load_dataset("HuggingFaceH4/MATH-500", split="test")
    lookup = {}
    for idx, row in enumerate(dataset):
        level_raw = row.get("level")
        try:
            level = int(str(level_raw).strip().split()[-1]) if level_raw is not None else None
        except (ValueError, IndexError):
            level = None
        lookup[idx] = level
    return lookup


def id_to_index(row_id):
    m = re.search(r"(\d+)$", row_id)
    return int(m.group(1)) if m else None


def main(trace_file, out_path):
    level_lookup = load_level_lookup()

    accuracy = defaultdict(lambda: {"correct": 0, "total": 0})
    bucket_counts = defaultdict(lambda: defaultdict(lambda: {"correct": 0, "incorrect": 0}))
    word_fracs = defaultdict(lambda: {"correct": [], "incorrect": []})
    never_explicit_counts = defaultdict(lambda: {"correct": 0, "incorrect": 0})
    unmapped = 0
    total = 0

    with open(trace_file) as f_in, open(out_path, "w") as f_out:
        for line in f_in:
            row = json.loads(line)
            reasoning = row.get("reasoning_text")
            gold = row.get("answer")
            correct = row.get("correct")
            row_id = row.get("id")
            if reasoning is None or gold is None or correct is None or row_id is None:
                continue

            idx = id_to_index(row_id)
            level = level_lookup.get(idx) if idx is not None else None
            tier = TIER_NAMES.get(level)
            if tier is None:
                unmapped += 1
                continue

            total += 1
            label = "correct" if correct else "incorrect"
            accuracy[tier]["total"] += 1
            if correct:
                accuracy[tier]["correct"] += 1

            bucket, cut_idx, n_sentences = bucket_trace(reasoning, gold)
            bucket_counts[tier][bucket][label] += 1

            sentences = split_sentences(reasoning)
            if bucket == "never_explicit" or not sentences:
                never_explicit_counts[tier][label] += 1
                kept_text = reasoning
                frac_word = 1.0
                never_explicit = True
            else:
                kept_sentences = sentences[:cut_idx + 1]
                kept_text = " ".join(kept_sentences)
                orig_words = len(reasoning.split())
                frac_word = (len(kept_text.split()) / orig_words) if orig_words > 0 else 1.0
                word_fracs[tier][label].append(frac_word)
                never_explicit = False

            out_row = {
                "id": row_id,
                "correct": correct,
                "answer": gold,
                "difficulty_tier": tier,
                "math_level": level,
                "never_explicit": never_explicit,
                "cut_sentence_idx": cut_idx,
                "total_sentences": n_sentences,
                "frac_words_kept": round(frac_word, 4),
                "adaptive_truncated_text": kept_text,
            }
            f_out.write(json.dumps(out_row) + "\n")

    print(f"Analyzed {total} traces from {trace_file} ({unmapped} skipped: no difficulty level found)\n")

    print("=" * 78)
    print("ACCURACY BY DIFFICULTY TIER")
    print("=" * 78)
    print(f"{'Tier':<10} {'Correct':>8} {'Total':>8} {'% Correct':>10}")
    for tier in TIER_ORDER:
        a = accuracy[tier]
        pct = 100 * a["correct"] / a["total"] if a["total"] else 0.0
        print(f"{tier:<10} {a['correct']:>8} {a['total']:>8} {pct:>9.1f}%")

    print()
    print("=" * 78)
    print("ANSWER POSITION, BY DIFFICULTY TIER  (fixed matching: no numeric substring false positives)")
    print("=" * 78)
    for tier in TIER_ORDER:
        print(f"\n--- {tier} ---")
        print(f"{'Bucket':<20} {'Correct':>8} {'Incorrect':>10} {'Total':>7} {'% Correct':>10}")
        print("-" * 60)
        for bucket, label_display in [
            ("early_or_middle", "Early/intermediate"),
            ("last_two", "Last 2 sentences"),
            ("never_explicit", "Never explicit"),
        ]:
            c = bucket_counts[tier][bucket]["correct"]
            i = bucket_counts[tier][bucket]["incorrect"]
            t = c + i
            pct = (100 * c / t) if t > 0 else 0.0
            print(f"{label_display:<20} {c:>8} {i:>10} {t:>7} {pct:>9.1f}%")

    print()
    print("=" * 78)
    print("ADAPTIVE-CUT COMPRESSION, BY DIFFICULTY TIER  (excluding never-explicit traces)")
    print("=" * 78)
    print(f"{'Tier':<10} {'Group':<12} {'N':>5} {'Avg % kept (words)':>20}")
    print("-" * 52)
    for tier in TIER_ORDER:
        for label in ["correct", "incorrect"]:
            vals = word_fracs[tier][label]
            n = len(vals)
            avg = 100 * sum(vals) / n if n else 0
            print(f"{tier:<10} {label:<12} {n:>5} {avg:>19.1f}%")

    print(f"\nTruncated traces (tagged with difficulty_tier) saved to {out_path}")
    print(
        "\nNext step: run score_by_difficulty.py on this file to see whether the "
        "adaptive-cut signal-retention result holds up specifically on Hard problems."
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--trace-file", required=True)
    parser.add_argument("--out", default="adaptive_cut_traces_by_difficulty.jsonl")
    args = parser.parse_args()
    main(args.trace_file, args.out)
