"""Gold-free position bucketing + adaptive cut + (re)labelling. Works on both
trace schemas (reasoning teacher: predicted_answer/extraction_method; instruct
teacher: pred/level). See RESULTS.md for what this produced and GLOSSARY.md for
the definitions. Run from the repo root:

    python analysis/analyze_traces_v3.py --trace-file data/traces_X.jsonl --out data/cut_X.jsonl --with-difficulty
"""
import argparse
import json
import os
import re
import sys
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common.answer_matching import (  # noqa: E402
    answers_match, cut_target, is_restatement, normalize_for_search,
    sentence_mentions_answer, split_sentences, strip_wrappers,
)

TIER_NAMES = {1: "Easy", 2: "Easy", 3: "Medium", 4: "Hard", 5: "Hard"}
TIER_ORDER = ["Easy", "Medium", "Hard"]


def load_level_lookup():
    from datasets import load_dataset
    ds = load_dataset("HuggingFaceH4/MATH-500", split="test")
    out = {}
    for i, row in enumerate(ds):
        try:
            out[i] = int(str(row.get("level")).strip().split()[-1])
        except (ValueError, IndexError):
            out[i] = None
    return out


def pct(a, b):
    return 100.0 * a / b if b else 0.0


def main(trace_file, out_path, with_difficulty):
    levels = load_level_lookup() if with_difficulty else None
    rows_out = []
    flips_up = flips_down = n_relabelled = 0
    acc = defaultdict(lambda: [0, 0])
    buckets = defaultdict(lambda: defaultdict(lambda: [0, 0]))
    fracs = {"correct": [], "incorrect": []}
    never = [0, 0]

    with open(trace_file) as f:
        for line in f:
            r = json.loads(line)
            if "predicted_answer" not in r and "pred" in r:
                r["predicted_answer"] = r["pred"]
                r["extraction_method"] = "boxed" if r["pred"] is not None else "none"
            text, gold, rid = r.get("reasoning_text"), r.get("answer"), r.get("id")
            if text is None or gold is None or rid is None:
                continue
            correct = bool(r.get("correct"))
            if "predicted_answer" in r:
                new = answers_match(r["predicted_answer"], gold, r.get("extraction_method"))
                n_relabelled += 1
                if new and not correct:
                    flips_up += 1
                if correct and not new:
                    flips_down += 1
                correct = new

            tier = "All"
            if levels is not None:
                lv = None
                if r.get("level") is not None:
                    mm = re.search(r"(\d)", str(r["level"]))
                    lv = int(mm.group(1)) if mm else None
                else:
                    m = re.search(r"(\d+)$", rid)
                    lv = levels.get(int(m.group(1))) if m else None
                tier = TIER_NAMES.get(lv)
                if tier is None:
                    continue

            sents = split_sentences(text)
            target = cut_target(r, correct, gold)
            idx = None
            if sents and target is not None:
                ng = normalize_for_search(strip_wrappers(target))
                prob = r.get("problem", "")
                warmup = max(3, len(sents) // 10)
                for i, sn in enumerate(sents):
                    if i < warmup or is_restatement(sn, prob):
                        continue
                    if sentence_mentions_answer(normalize_for_search(sn), ng):
                        idx = i
                        break
            if idx is None:
                bucket, kept, frac = "never_explicit", text, 1.0
                never[0 if correct else 1] += 1
            else:
                bucket = "last_two" if idx >= len(sents) - 2 else "early_or_middle"
                kept = " ".join(sents[: idx + 1])
                frac = len(kept.split()) / max(1, len(text.split()))
                fracs["correct" if correct else "incorrect"].append(frac)

            acc[tier][0] += int(correct)
            acc[tier][1] += 1
            buckets[tier][bucket][0 if correct else 1] += 1
            rows_out.append({
                "id": rid, "sample_idx": r.get("sample_idx"), "correct": correct, "answer": gold, "cut_target": target, "predicted_answer": r.get("predicted_answer"), "difficulty_tier": tier,
                "never_explicit": idx is None, "cut_sentence_idx": idx,
                "total_sentences": len(sents), "frac_words_kept": round(frac, 4),
                "adaptive_truncated_text": kept,
            })

    with open(out_path, "w") as f:
        for r in rows_out:
            f.write(json.dumps(r) + "\n")

    total = sum(v[1] for v in acc.values())
    print(f"Analyzed {total} traces from {trace_file}")
    if n_relabelled:
        print(f"Correctness recomputed for {n_relabelled} traces: "
              f"{flips_up} incorrect->correct, {flips_down} correct->incorrect")
    print("\nACCURACY (trace level)")
    for t in (TIER_ORDER if with_difficulty else ["All"]):
        c, n = acc[t]
        print(f"  {t:<8} {c}/{n} = {pct(c, n):.1f}%")

    print("\nANSWER POSITION (first real statement of the MODEL'S OWN final answer, gold-free)")
    print(f"  {'Tier':<8}{'Bucket':<18}{'Correct':>8}{'Incorrect':>10}{'Total':>7}{'%Correct':>10}")
    allb = defaultdict(lambda: [0, 0])
    for t in (TIER_ORDER if with_difficulty else ["All"]):
        for b in ["early_or_middle", "last_two", "never_explicit"]:
            c, i = buckets[t][b]
            allb[b][0] += c
            allb[b][1] += i
            print(f"  {t:<8}{b:<18}{c:>8}{i:>10}{c + i:>7}{pct(c, c + i):>9.1f}%")
    if with_difficulty:
        for b in ["early_or_middle", "last_two", "never_explicit"]:
            c, i = allb[b]
            print(f"  {'ALL':<8}{b:<18}{c:>8}{i:>10}{c + i:>7}{pct(c, c + i):>9.1f}%")

    print("\nADAPTIVE CUT: average % of words kept (traces with a real cut point only)")
    for label in ["correct", "incorrect"]:
        v = fracs[label]
        print(f"  {label:<10} N={len(v):<4} {100 * sum(v) / len(v) if v else 0:.1f}% kept")
    print(f"\nNever-explicit traces (kept in full, excluded above): correct={never[0]} incorrect={never[1]}")
    print(f"Cut traces saved to {out_path}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--trace-file", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--with-difficulty", action="store_true")
    a = ap.parse_args()
    main(a.trace_file, a.out, a.with_difficulty)
