"""Do the two teachers' trace files cover the SAME MATH-500 problems? Prints the overlap, the difficulty
mix of each set, and (on the overlap, if any) a like-for-like accuracy comparison.

Why: the instruct and reasoning comparisons in RESULTS.md are only 'same problems' if the id sets match.
Ids look like 'math500-267' (instruct) and 'math500-reasoning-4' (reasoning); the trailing number is the
MATH-500 test index.

Usage (repo root), using the cut files (they carry the recomputed `correct`):
    python diagnostics/compare_problem_sets.py --a data/cut_instruct_final.jsonl --b data/cut_reasoning_final.jsonl
"""
import argparse
import collections
import json
import re


def load(path):
    per = collections.defaultdict(list)
    for line in open(path):
        r = json.loads(line)
        m = re.search(r"(\d+)$", r["id"])
        per[int(m.group(1))].append(bool(r["correct"]))
    return per


def levels():
    try:
        from datasets import load_dataset
        ds = load_dataset("HuggingFaceH4/MATH-500", split="test")
        return {i: int(str(r["level"]).split()[-1]) for i, r in enumerate(ds)}
    except Exception as e:  # offline: skip tier info
        print("(could not load MATH-500 levels:", e, ")")
        return {}


def tier(l):
    return None if l is None else ("Easy" if l <= 2 else "Medium" if l == 3 else "Hard")


def main(a_path, b_path):
    a, b = load(a_path), load(b_path)
    lv = levels()
    both = sorted(set(a) & set(b))
    print(f"A = {a_path}: {len(a)} problems   B = {b_path}: {len(b)} problems")
    print(f"Overlap: {len(both)} problems   A only: {len(set(a) - set(b))}   B only: {len(set(b) - set(a))}")
    for name, s in (("A", a), ("B", b)):
        mix = collections.Counter(tier(lv.get(i)) for i in s)
        print(f"  difficulty mix of {name} (problems): {dict(mix)}")
    if both:
        for name, s in (("A", a), ("B", b)):
            ok = sum(sum(s[i]) for i in both)
            n = sum(len(s[i]) for i in both)
            print(f"  accuracy of {name} on the overlap: {ok}/{n} = {100 * ok / n:.1f}%")
    else:
        print("No overlapping problems: any accuracy gap between the two files mixes teacher with problem difficulty.")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--a", required=True)
    p.add_argument("--b", required=True)
    a = p.parse_args()
    main(a.a, a.b)
