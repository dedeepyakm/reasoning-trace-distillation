"""
Adaptive answer-position truncation -- no GPU/model needed, pure text analysis.

For each trace, finds the first sentence where the gold answer is explicitly
stated (same logic as answer_position_analysis.py), then truncates the trace
to end right after that sentence -- discarding everything past the point
where the model already committed to the answer.

Traces that never state the answer explicitly are kept in full (nothing to
cut to).

Reports how much compression this achieves, split by correct vs incorrect,
and saves the truncated traces to a new file for further scoring later.

Usage:
    python3 adaptive_cut_analysis.py --trace-file traces_teacher_k5.jsonl --out adaptive_cut_traces.jsonl
"""

import argparse
import json
import re


def normalize_for_search(text):
    text = text.lower().strip()
    text = text.replace("$", "").replace(",", "")
    text = re.sub(r"\s+", " ", text)
    return text


def split_sentences(text):
    parts = re.split(r'(?<=[.?!])\s+', text.strip())
    return [p for p in parts if p]


def find_first_mention_sentence(sentences, gold_answer):
    norm_gold = normalize_for_search(gold_answer)
    if not norm_gold:
        return None
    for i, sent in enumerate(sentences):
        if norm_gold in normalize_for_search(sent):
            return i
    return None


def main(trace_file, out_path):
    total = 0
    never_explicit_count = 0
    word_fracs = {"correct": [], "incorrect": []}
    sent_fracs = {"correct": [], "incorrect": []}

    with open(trace_file) as f_in, open(out_path, "w") as f_out:
        for line in f_in:
            row = json.loads(line)
            reasoning = row.get("reasoning_text")
            gold = row.get("answer")
            correct = row.get("correct")
            if reasoning is None or gold is None or correct is None:
                continue

            sentences = split_sentences(reasoning)
            if not sentences:
                continue

            idx = find_first_mention_sentence(sentences, gold)
            if idx is None:
                kept_sentences = sentences  # can't adaptively cut, keep everything
                never_explicit = True
                never_explicit_count += 1
            else:
                kept_sentences = sentences[:idx + 1]
                never_explicit = False

            kept_text = " ".join(kept_sentences)
            frac_sent = len(kept_sentences) / len(sentences)
            orig_words = len(reasoning.split())
            frac_word = (len(kept_text.split()) / orig_words) if orig_words > 0 else 1.0

            label = "correct" if correct else "incorrect"
            word_fracs[label].append(frac_word)
            sent_fracs[label].append(frac_sent)
            total += 1

            out_row = {
                "id": row.get("id"),
                "correct": correct,
                "answer": gold,
                "never_explicit": never_explicit,
                "cut_sentence_idx": idx,
                "total_sentences": len(sentences),
                "frac_words_kept": round(frac_word, 4),
                "adaptive_truncated_text": kept_text,
            }
            f_out.write(json.dumps(out_row) + "\n")

    print(f"Processed {total} traces from {trace_file}\n")
    print(f"Traces with no explicit answer (kept in full, no cut possible): "
          f"{never_explicit_count} ({100 * never_explicit_count / total:.1f}%)\n")

    print(f"{'Group':<12} {'N':>5} {'Avg % of trace kept (words)':>28} {'Avg % kept (sentences)':>24}")
    print("-" * 72)
    for label in ["correct", "incorrect"]:
        n = len(word_fracs[label])
        avg_word = 100 * sum(word_fracs[label]) / n if n else 0
        avg_sent = 100 * sum(sent_fracs[label]) / n if n else 0
        print(f"{label:<12} {n:>5} {avg_word:>27.1f}% {avg_sent:>23.1f}%")

    all_word = word_fracs["correct"] + word_fracs["incorrect"]
    print(f"\nOverall average compression: kept {100 * sum(all_word) / len(all_word):.1f}% "
          f"of trace length (by words), vs. 100% in the original.")
    print(f"Truncated traces saved to {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--trace-file", required=True)
    parser.add_argument("--out", default="adaptive_cut_traces.jsonl")
    args = parser.parse_args()
    main(args.trace_file, args.out)
