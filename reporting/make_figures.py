"""Make figures straight from a cut file (and optionally a score file). Nothing is typed in by hand.

Every number on a figure is computed here from the file you pass in, and each figure is stamped
with its source file. The same numbers are appended to logs/RESULT_LOG.csv.

Usage:
  python reporting/make_figures.py --cut-file data/cut_reasoning_final.jsonl \
      --score-file data/score_reasoning_final.json --label reasoning --outdir data/figures
"""
import argparse
import json
import math
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common.run_logging import log_result  # noqa: E402

SCRIPT = "reporting/make_figures.py"
GOOD, BAD, NEUTRAL = "#2a7f62", "#c0392b", "#4a6fa5"
TIERS = ["Easy", "Medium", "Hard"]


def wilson(k, n, z=1.96):
    if n == 0:
        return 0.0, 0.0
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return max(0.0, c - h), min(1.0, c + h)


def load(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]


def stamp(fig, source, extra=""):
    fig.text(0.01, 0.005, "source: {}   script: {}   {}".format(source, SCRIPT, extra),
             fontsize=6, color="#666666")


def fig_accuracy_by_tier(rows, label, source, outdir):
    tiers = [t for t in TIERS if any(r.get("difficulty_tier") == t for r in rows)]
    if not tiers:
        print("No difficulty tiers in this cut file (run analysis with --with-difficulty); skipping tier figure.")
        return
    fig, ax = plt.subplots(figsize=(5.5, 4))
    for i, t in enumerate(tiers):
        rs = [r for r in rows if r.get("difficulty_tier") == t]
        k, n = sum(bool(r["correct"]) for r in rs), len(rs)
        lo, hi = wilson(k, n)
        ax.bar(i, 100 * k / n, color=NEUTRAL, yerr=[[max(0.0, 100 * k / n - 100 * lo)], [max(0.0, 100 * hi - 100 * k / n)]],
               capsize=4)
        ax.text(i, 3, "{}/{}".format(k, n), ha="center", color="white", fontsize=9)
        log_result(label, "accuracy_" + t, "{}/{} = {:.1f}%".format(k, n, 100 * k / n), source, SCRIPT)
    ax.set_xticks(range(len(tiers)))
    ax.set_xticklabels(tiers)
    ax.set_ylim(0, 105)
    ax.set_ylabel("% of traces correct (95% Wilson interval)")
    ax.set_title("{}: accuracy by difficulty".format(label))
    stamp(fig, source)
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    fig.savefig(os.path.join(outdir, "{}_accuracy_by_tier.png".format(label)), dpi=160)
    plt.close(fig)


def bucket(r):
    if r.get("never_explicit"):
        return "never explicit"
    idx, tot = r.get("cut_sentence_idx"), r.get("total_sentences")
    if idx is None or tot is None:
        return "never explicit"
    return "last two sentences" if idx >= tot - 2 else "before last two"


def fig_position(rows, label, source, outdir):
    names = ["before last two", "last two sentences", "never explicit"]
    fig, ax = plt.subplots(figsize=(6, 4))
    for i, b in enumerate(names):
        rs = [r for r in rows if bucket(r) == b]
        c = sum(bool(r["correct"]) for r in rs)
        w = len(rs) - c
        ax.bar(i, c, color=GOOD, label="correct" if i == 0 else None)
        ax.bar(i, w, bottom=c, color=BAD, label="incorrect" if i == 0 else None)
        pct = 100 * c / len(rs) if rs else 0
        ax.text(i, c + w + 1, "n={}  {:.0f}% correct".format(len(rs), pct), ha="center", fontsize=9)
        log_result(label, "position_" + b.replace(" ", "_"),
                   "n={} correct={} incorrect={}".format(len(rs), c, w), source, SCRIPT)
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(names)
    ax.set_ylabel("number of traces")
    ax.set_ylim(0, ax.get_ylim()[1] * 1.12)
    ax.set_title("{}: where the model first states its own answer".format(label))
    ax.legend()
    stamp(fig, source)
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    fig.savefig(os.path.join(outdir, "{}_answer_position.png".format(label)), dpi=160)
    plt.close(fig)


def fig_words_kept(rows, label, source, outdir):
    cut = [r for r in rows if not r.get("never_explicit") and r.get("frac_words_kept") is not None]
    cor = [100 * r["frac_words_kept"] for r in cut if r["correct"]]
    inc = [100 * r["frac_words_kept"] for r in cut if not r["correct"]]
    fig, ax = plt.subplots(figsize=(6, 4))
    bins = list(range(0, 101, 10))
    ax.hist([cor, inc], bins=bins, color=[GOOD, BAD], label=["correct (N={})".format(len(cor)),
                                                             "incorrect (N={})".format(len(inc))])
    ax.set_xlabel("% of words kept after the cut")
    ax.set_ylabel("number of traces")
    ax.set_title("{}: how much of each trace survives the cut".format(label))
    ax.legend()
    stamp(fig, source, "traces with a cut point only")
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    fig.savefig(os.path.join(outdir, "{}_words_kept.png".format(label)), dpi=160)
    plt.close(fig)
    for name, v in (("correct", cor), ("incorrect", inc)):
        if v:
            log_result(label, "words_kept_" + name, "{:.1f}% (N={})".format(sum(v) / len(v), len(v)),
                       source, SCRIPT)


def flatten(d, prefix=""):
    out = {}
    if isinstance(d, dict):
        for k, v in d.items():
            out.update(flatten(v, prefix + "/" + str(k) if prefix else str(k)))
    elif isinstance(d, list):
        for i, v in enumerate(d):
            out.update(flatten(v, "{}[{}]".format(prefix, i)))
    else:
        out[prefix] = d
    return out


def fig_retention(score_path, label, outdir):
    """Retention = cut separation / full separation, computed here from the score file.
    The score file has no bootstrap interval, so none is drawn (the interval printed by the
    scoring script is in the run log)."""
    with open(score_path, encoding="utf-8") as f:
        flat = flatten(json.load(f))
    need = ["cut/separation", "full/separation", "cut/n_correct", "cut/n_incorrect"]
    missing = [k for k in need if k not in flat]
    if missing:
        print("Score file {} lacks {}; retention figure skipped (nothing invented).".format(score_path, missing))
        print("Keys found: {}".format(list(flat)))
        return
    cut, full = flat["cut/separation"], flat["full/separation"]
    retention = 100 * cut / full
    nc, ni = flat["cut/n_correct"], flat["cut/n_incorrect"]
    fig, ax = plt.subplots(figsize=(5.5, 4))
    ax.bar([0, 1], [full, cut], color=[NEUTRAL, GOOD])
    for x, v in ((0, full), (1, cut)):
        ax.text(x, v + 0.03 * full, "{:.3f}".format(v), ha="center", fontsize=10)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["full trace", "cut trace"])
    ax.set_ylim(0, full * 1.2)
    ax.set_ylabel("separation = mean logp(correct) - mean logp(incorrect)")
    ax.set_title("{}: retention {:.1f}% ({} correct / {} incorrect scored)".format(label, retention, nc, ni),
                 fontsize=10)
    stamp(fig, score_path, "retention = cut / full")
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    fig.savefig(os.path.join(outdir, "{}_retention.png".format(label)), dpi=160)
    plt.close(fig)
    log_result(label, "retention", "{:.1f}%".format(retention), score_path, SCRIPT,
               "cut sep {:.3f} vs full sep {:.3f}; scored {} correct / {} incorrect".format(cut, full, nc, ni))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cut-file", required=True)
    ap.add_argument("--score-file")
    ap.add_argument("--label", required=True)
    ap.add_argument("--outdir", default="data/figures")
    a = ap.parse_args()
    os.makedirs(a.outdir, exist_ok=True)
    rows = load(a.cut_file)
    print("Loaded {} traces from {}".format(len(rows), a.cut_file))
    fig_accuracy_by_tier(rows, a.label, a.cut_file, a.outdir)
    fig_position(rows, a.label, a.cut_file, a.outdir)
    fig_words_kept(rows, a.label, a.cut_file, a.outdir)
    if a.score_file and os.path.exists(a.score_file):
        fig_retention(a.score_file, a.label, a.outdir)
    print("Figures saved in {}".format(a.outdir))


if __name__ == "__main__":
    main()
