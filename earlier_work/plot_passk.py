#!/usr/bin/env python3
"""
Plot pass@k curves for all trace files. Uses the unbiased estimator
(Chen et al. 2021): pass@k = 1 - C(n-c,k)/C(n,k)

    python plot_passk.py traces_*.jsonl
"""
import json, sys, glob
from collections import defaultdict
from math import comb
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def pass_at_k(n, c, k):
    if k > n:
        return None
    if n - c < k:
        return 1.0
    return 1.0 - comb(n - c, k) / comb(n, k)


def curve(path):
    rows = [json.loads(l) for l in open(path)]
    model = rows[0].get("model", path).split("/")[-1]
    by = defaultdict(list)
    for r in rows:
        by[r["id"]].append(r)
    n_per = min(len(v) for v in by.values())

    ks, scores = [], []
    for k in range(1, n_per + 1):
        vals = [pass_at_k(len(v), sum(1 for s in v if s["correct"]), k)
                for v in by.values()]
        vals = [v for v in vals if v is not None]
        ks.append(k)
        scores.append(100 * sum(vals) / len(vals))

    # share of problems solved on some but not all samples
    some = sum(1 for v in by.values()
               if 0 < sum(1 for s in v if s["correct"]) < len(v)) / len(by)
    return model, ks, scores, some, len(by)


def main():
    paths = []
    for a in sys.argv[1:]:
        paths.extend(glob.glob(a))
    if not paths:
        print("usage: python plot_passk.py traces_*.jsonl")
        return

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))
    rows = []
    for p in sorted(paths):
        model, ks, scores, some, n = curve(p)
        ax1.plot(ks, scores, marker="o", label=f"{model} (n={n})")
        rows.append((model, ks[-1], scores[0], scores[-1], some))

    ax1.set_xlabel("k (samples per problem)")
    ax1.set_ylabel("pass@k  (%)")
    ax1.set_title("pass@k vs k")
    ax1.grid(alpha=.3)
    ax1.legend(fontsize=8)
    ax1.set_xticks(range(1, max(r[1] for r in rows) + 1))

    # gain over pass@1 -- shows how much is reliability vs capability
    for p in sorted(paths):
        model, ks, scores, _, _ = curve(p)
        ax2.plot(ks, [s - scores[0] for s in scores], marker="o", label=model)
    ax2.set_xlabel("k (samples per problem)")
    ax2.set_ylabel("gain over pass@1  (percentage points)")
    ax2.set_title("How much does extra sampling recover?")
    ax2.grid(alpha=.3)
    ax2.set_xticks(range(1, max(r[1] for r in rows) + 1))

    plt.tight_layout()
    plt.savefig("passk_curves.png", dpi=150)
    print("wrote passk_curves.png\n")

    print(f"{'model':<34}{'max k':>6}{'pass@1':>9}{'pass@maxk':>11}{'sometimes':>11}")
    for m, mk, p1, pk, some in rows:
        print(f"{m:<34}{mk:>6}{p1:>8.1f}%{pk:>10.1f}%{some:>10.0%}")
    print("\n'sometimes' = share of problems solved on some but not all samples.")
    print("pass@k rises only from these, so it predicts how steep each curve is.")


if __name__ == "__main__":
    main()
