#!/usr/bin/env python3
"""
Analysis for the pilot run. CPU-only, fast, re-runnable without regenerating.

  python analyze_pilot.py traces_Qwen2-5-7B-Instruct_math500.jsonl \
                          traces_gpt-oss-20b_math500.jsonl

Produces four analyses per model, plus a cross-model comparison:
  1. Accuracy                      -- baseline sanity
  2. Calibration / confirmation bias -- does confidence separate correct from wrong?
  3. Trace length                  -- the Qwen vs gpt-oss token question
  4. Confidence vs difficulty      -- does the signal track real difficulty?
"""

import argparse, json, sys
import numpy as np
import pandas as pd
from scipy import stats

CONF_FIELDS = [
    ("logp_r_mean", "reasoning confidence (length-normalised)"),
    ("logp_y_mean", "answer confidence (length-normalised)"),
    ("logp_r_sum",  "reasoning confidence (raw sum)"),
    ("picsar_sum",  "PiCSAR joint score (raw sum)"),
]


def load(path):
    rows = [json.loads(l) for l in open(path) if l.strip()]
    df = pd.DataFrame(rows)
    df["picsar_mean"] = df["logp_r_mean"] + df["logp_y_mean"]
    return df


def cohens_d(a, b):
    na, nb = len(a), len(b)
    if na < 2 or nb < 2:
        return float("nan")
    pooled = np.sqrt(((na - 1) * np.var(a, ddof=1) +
                      (nb - 1) * np.var(b, ddof=1)) / (na + nb - 2))
    return (np.mean(a) - np.mean(b)) / pooled if pooled > 0 else float("nan")


def auroc(scores, labels):
    """P(score of a correct sample > score of an incorrect one). 0.5 = no signal."""
    pos = np.asarray(scores)[np.asarray(labels, dtype=bool)]
    neg = np.asarray(scores)[~np.asarray(labels, dtype=bool)]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    u = stats.mannwhitneyu(pos, neg, alternative="two-sided").statistic
    return u / (len(pos) * len(neg))


def analyse(df, name):
    print("\n" + "=" * 74)
    print(f"MODEL: {name}    ({df['id'].nunique()} problems, {len(df)} samples)")
    print("=" * 74)

    # ---- 1. accuracy --------------------------------------------------------
    pass1 = df.groupby("id")["correct"].first().mean()
    passk = df.groupby("id")["correct"].any().mean()
    print(f"\n[1] ACCURACY")
    print(f"    pass@1 (first sample) : {pass1:6.1%}")
    print(f"    pass@k (any sample)   : {passk:6.1%}")
    print(f"    sample-level accuracy : {df['correct'].mean():6.1%}")
    if pass1 > 0.95:
        print("    ! near-ceiling: too easy to show a teacher-student gap")
    if pass1 < 0.10:
        print("    ! near-floor: teacher traces will be unreliable supervision")

    # ---- 2. calibration / confirmation bias ---------------------------------
    print(f"\n[2] CALIBRATION  (confirmation bias check)")
    print("    Does confidence separate correct from incorrect answers?")
    print(f"    {'signal':<42}{'correct':>9}{'wrong':>9}{'d':>7}{'AUROC':>7}{'p':>9}")
    for field, label in CONF_FIELDS + [("picsar_mean", "PiCSAR joint (length-normalised)")]:
        if field not in df or df[field].isna().all():
            continue
        sub = df.dropna(subset=[field])
        c = sub.loc[sub["correct"], field].values
        w = sub.loc[~sub["correct"], field].values
        if len(c) < 2 or len(w) < 2:
            print(f"    {label:<42}  (insufficient samples in one class)")
            continue
        t, p = stats.ttest_ind(c, w, equal_var=False)
        print(f"    {label:<42}{np.mean(c):9.3f}{np.mean(w):9.3f}"
              f"{cohens_d(c, w):7.2f}{auroc(sub[field].values, sub['correct'].values):7.2f}"
              f"{p:9.1e}")
    print("    Reading: AUROC ~0.5 and d ~0 => confidence carries no information about")
    print("    correctness, i.e. the model is confident whether right or wrong.")
    print("    That is the confirmation-bias signature, and it would undermine any")
    print("    curriculum ordered by confidence.")

    # ---- 3. trace length ----------------------------------------------------
    print(f"\n[3] TRACE LENGTH")
    print(f"    mean total tokens     : {df['n_tokens_total'].mean():8.0f}")
    print(f"    median total tokens   : {df['n_tokens_total'].median():8.0f}")
    print(f"    mean reasoning tokens : {df['n_tokens_reasoning'].mean():8.0f}")
    if df["correct"].nunique() > 1:
        lc = df.loc[df["correct"], "n_tokens_total"]
        lw = df.loc[~df["correct"], "n_tokens_total"]
        print(f"    correct traces        : {lc.mean():8.0f} tokens")
        print(f"    incorrect traces      : {lw.mean():8.0f} tokens")
        if lw.mean() > 0:
            print(f"    ratio wrong/correct   : {lw.mean()/max(lc.mean(),1):8.2f}x")
        print("    (literature expects incorrect traces to run longer)")

    # ---- 4. confidence vs ground-truth difficulty ---------------------------
    print(f"\n[4] CONFIDENCE vs GROUND-TRUTH DIFFICULTY")
    if "level" in df and df["level"].notna().any():
        d = df.dropna(subset=["level"]).copy()
        d["level_num"] = pd.to_numeric(
            d["level"].astype(str).str.extract(r"(\d+)")[0], errors="coerce")
        d = d.dropna(subset=["level_num"])
        if len(d) > 10:
            for field, label in [("logp_r_mean", "reasoning confidence"),
                                 ("picsar_mean", "PiCSAR joint")]:
                sub = d.dropna(subset=[field])
                rho, p = stats.spearmanr(sub[field], sub["level_num"])
                print(f"    {label:<28} rho={rho:+.3f}  p={p:.1e}")
            print("\n    accuracy and confidence by level:")
            g = d.groupby("level_num").agg(
                n=("correct", "size"), acc=("correct", "mean"),
                conf=("logp_r_mean", "mean"), tokens=("n_tokens_total", "mean"))
            print(g.to_string(float_format=lambda x: f"{x:8.3f}"))
            print("\n    Expect: negative rho (harder level => lower confidence).")
            print("    Near-zero rho falsifies the 'confidence = difficulty'")
            print("    assumption that Approach C rests on.")
        else:
            print("    not enough levelled rows")
    else:
        print("    no difficulty labels in this dataset (use MATH500 for this analysis)")

    return {
        "model": name, "pass@1": pass1, "pass@k": passk,
        "mean_tokens": df["n_tokens_total"].mean(),
        "auroc_reasoning_conf": auroc(df["logp_r_mean"].fillna(0).values,
                                      df["correct"].values),
        "auroc_picsar": auroc(df["picsar_mean"].fillna(0).values,
                              df["correct"].values),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("files", nargs="+")
    args = ap.parse_args()

    summaries = []
    for path in args.files:
        df = load(path)
        name = df["model"].iloc[0] if "model" in df else path
        summaries.append(analyse(df, name))

    if len(summaries) > 1:
        print("\n" + "=" * 74)
        print("CROSS-MODEL COMPARISON")
        print("=" * 74)
        s = pd.DataFrame(summaries)
        print(s.to_string(index=False, float_format=lambda x: f"{x:.3f}"))
        print("\nKey questions this table answers:")
        print("  * Which model's confidence better predicts correctness? (AUROC)")
        print("  * How large is the token-efficiency gap? (mean_tokens)")
        print("  * Is accuracy comparable despite that gap?  (pass@1)")
        print("\nNote: compare *length-normalised* confidence across models only;")
        print("raw sums are confounded by trace length.")


if __name__ == "__main__":
    main()
