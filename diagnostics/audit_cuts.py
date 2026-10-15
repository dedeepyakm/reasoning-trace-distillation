"""Manual audit of adaptive cuts: samples N random cut traces (those with a real cut point)
and writes a CSV you fill in by hand, so we can report the PRECISION of the cut
(share of cuts that land on the model's real conclusion).

For each sampled trace it prints the problem, the model's final answer, and the last
~400 characters of the kept text. Label each one in the CSV column `label`:
    ok        kept text ends on the real conclusion (answer stated as a result)
    early     answer appears but only as an intermediate / listed number / restatement
    other     anything else

Usage (from repo root):
    python diagnostics/audit_cuts.py --cut-file data/cut_X.jsonl --n 30 --out data/audit_X.csv
Then:
    python diagnostics/audit_cuts.py --score data/audit_X.csv
"""
import argparse
import csv
import json
import random


def sample(cut_file, n, out, seed):
    rows = [json.loads(l) for l in open(cut_file)]
    rows = [r for r in rows if not r.get("never_explicit")]
    random.Random(seed).shuffle(rows)
    rows = rows[:n]
    with open(out, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["id", "sample_idx", "correct", "gold", "cut_target", "frac_words_kept", "kept_tail", "label"])
        for r in rows:
            w.writerow([r["id"], r.get("sample_idx"), r["correct"], r["answer"], r.get("cut_target"),
                        r["frac_words_kept"], r["adaptive_truncated_text"][-400:].replace("\n", " "), ""])
    for i, r in enumerate(rows, 1):
        print("=" * 78)
        print(f"[{i}] id={r['id']} sample={r.get('sample_idx')} correct={r['correct']} "
              f"cut_target={r.get('cut_target')} kept={r['frac_words_kept']}")
        print("...", r["adaptive_truncated_text"][-400:])
    print(f"\nWrote {out}. Fill the `label` column (ok / early / other), then run with --score.")


def score(path):
    labs = [r["label"].strip().lower() for r in csv.DictReader(open(path, encoding="utf-8")) if r["label"].strip()]
    n = len(labs)
    if not n:
        print("No labels filled in yet.")
        return
    ok = labs.count("ok")
    print(f"Labelled {n}: ok={ok} early={labs.count('early')} other={labs.count('other')}  ->  precision {100 * ok / n:.0f}%")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--cut-file")
    p.add_argument("--n", type=int, default=30)
    p.add_argument("--out", default="audit.csv")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--score")
    a = p.parse_args()
    score(a.score) if a.score else sample(a.cut_file, a.n, a.out, a.seed)
