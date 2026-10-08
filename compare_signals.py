#!/usr/bin/env python3
"""
Head-to-head comparison of candidate DIFFICULTY SIGNALS.

Answers the standing question: which signal should order the curriculum?
Uses only files already produced -- no GPU, no regeneration.

Signals compared (all correlated against MATH-500's human difficulty level):
  1. student entropy on the problem alone      H(y | x)      <- from info_gain.jsonl
  2. student reasoning confidence              log p(r | x)  <- from traces file
  3. teacher reasoning confidence              log p(r | x)  <- from traces file
  4. student pass-rate over k samples                        <- from traces file

Usage:
  python compare_signals.py --gain info_gain.jsonl \
      --student traces_Qwen2-5-1-5B-Instruct_math500.jsonl \
      --teacher traces_Qwen2-5-7B-Instruct_math500.jsonl
"""

import argparse, json, re
from collections import defaultdict
from scipy import stats


def level_num(lv):
    m = re.search(r"(\d+)", str(lv))
    return int(m.group(1)) if m else None


def load_traces(path):
    """Return {problem_id: {level, mean_conf, pass_rate}}"""
    rows = [json.loads(l) for l in open(path)]
    by = defaultdict(list)
    for r in rows:
        by[r["id"]].append(r)
    out = {}
    for pid, g in by.items():
        confs = [r["logp_r_mean"] for r in g if r.get("logp_r_mean") is not None]
        out[pid] = {
            "level": level_num(g[0].get("level")),
            "mean_conf": sum(confs) / len(confs) if confs else None,
            "pass_rate": sum(1 for r in g if r["correct"]) / len(g),
        }
    return out


def report(name, values, levels, higher_means_easier=True):
    pairs = [(v, l) for v, l in zip(values, levels)
             if v is not None and l is not None]
    if len(pairs) < 10:
        print(f"  {name:<38}  insufficient data ({len(pairs)})")
        return
    v = [a for a, _ in pairs]
    l = [b for _, b in pairs]
    rho, p = stats.spearmanr(v, l)
    sig = "yes" if p < 0.05 else "no"
    print(f"  {name:<38}{rho:>+8.3f}{p:>11.1e}{sig:>7}{len(pairs):>7}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gain", default="info_gain.jsonl")
    ap.add_argument("--student", required=True)
    ap.add_argument("--teacher", required=True)
    args = ap.parse_args()

    print("=" * 74)
    print("DIFFICULTY SIGNAL COMPARISON")
    print("  Spearman correlation of each signal against MATH-500 difficulty level.")
    print("  Expect NEGATIVE: harder problem -> lower confidence / lower pass rate.")
    print("  For entropy, expect POSITIVE: harder problem -> higher uncertainty.")
    print("=" * 74)
    print(f"\n  {'signal':<38}{'rho':>8}{'p':>11}{'sig':>7}{'n':>7}")

    # --- 1. student entropy on the problem alone -------------------------
    try:
        g = [json.loads(l) for l in open(args.gain)]
        # H_before depends only on the problem, so deduplicate by problem id
        seen = {}
        for r in g:
            if r["id"] not in seen:
                seen[r["id"]] = r
        H = [r["H_before"] for r in seen.values()]
        L = [level_num(r["level"]) for r in seen.values()]
        report("student entropy H(y|x)  [+ expected]", H, L)
    except FileNotFoundError:
        print(f"  (no {args.gain}; run info_gain.py first)")

    # --- 2/3. confidence signals -----------------------------------------
    st = load_traces(args.student)
    te = load_traces(args.teacher)

    report("student reasoning confidence",
           [v["mean_conf"] for v in st.values()],
           [v["level"] for v in st.values()])
    report("teacher reasoning confidence",
           [v["mean_conf"] for v in te.values()],
           [v["level"] for v in te.values()])

    # --- 4. pass rate (the field-standard difficulty measure) ------------
    report("student pass-rate over k samples",
           [v["pass_rate"] for v in st.values()],
           [v["level"] for v in st.values()])
    report("teacher pass-rate over k samples",
           [v["pass_rate"] for v in te.values()],
           [v["level"] for v in te.values()])

    # --- are the signals redundant? --------------------------------------
    print("\n" + "=" * 74)
    print("ARE THESE SIGNALS MEASURING THE SAME THING?")
    print("=" * 74)
    common = set(st) & set(te)
    if len(common) > 10:
        a = [st[p]["mean_conf"] for p in common]
        b = [st[p]["pass_rate"] for p in common]
        rho, p = stats.spearmanr(a, b)
        print(f"  student confidence  vs  student pass-rate : rho={rho:+.3f} p={p:.1e}")
        a = [te[p]["mean_conf"] for p in common]
        b = [te[p]["pass_rate"] for p in common]
        rho, p = stats.spearmanr(a, b)
        print(f"  teacher confidence  vs  teacher pass-rate : rho={rho:+.3f} p={p:.1e}")
        a = [st[p]["pass_rate"] for p in common]
        b = [te[p]["pass_rate"] for p in common]
        rho, p = stats.spearmanr(a, b)
        print(f"  student pass-rate   vs  teacher pass-rate : rho={rho:+.3f} p={p:.1e}")

    try:
        g = [json.loads(l) for l in open(args.gain)]
        seen = {}
        for r in g:
            if r["id"] not in seen:
                seen[r["id"]] = r
        ids = [p for p in seen if p in st]
        if len(ids) > 10:
            a = [seen[p]["H_before"] for p in ids]
            b = [st[p]["mean_conf"] for p in ids]
            rho, p = stats.spearmanr(a, b)
            print(f"  student entropy     vs  student confidence: rho={rho:+.3f} p={p:.1e}")
            b = [st[p]["pass_rate"] for p in ids]
            rho, p = stats.spearmanr(a, b)
            print(f"  student entropy     vs  student pass-rate : rho={rho:+.3f} p={p:.1e}")
    except FileNotFoundError:
        pass

    print("""
  Reading: strong correlation between two signals means they are largely
  redundant. Weak correlation means each carries information the other does
  not -- which would argue for combining them rather than choosing one.
""")


if __name__ == "__main__":
    main()
