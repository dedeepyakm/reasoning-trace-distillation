"""Scores ALL available cut traces (not one per problem) with the student model.
Aligns cut file and trace file on (id, sample_idx). Correct traces are randomly
subsampled (fixed seed); incorrect traces are all used up to --n-incorrect.
Reports mean logp of the gold answer for adaptive_cut vs full trace, and
separation, plus a per-trace paired difference with a bootstrap 95% interval.
"""
import argparse, json, os, random, sys
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common.scoring import (load_math500_lookup, get_problem_text,  # noqa: E402
                            score_gold_answer_logprob, STUDENT_MODEL)
from transformers import AutoTokenizer, AutoModelForCausalLM


def mean(v):
    return sum(v) / len(v) if v else float("nan")


def boot_ci(v, n=2000, seed=0):
    rnd = random.Random(seed)
    ms = sorted(mean([rnd.choice(v) for _ in v]) for _ in range(n))
    return ms[int(0.025 * n)], ms[int(0.975 * n)]


def main(a):
    full = {}
    with open(a.trace_file) as f:
        for line in f:
            r = json.loads(line)
            full[(r["id"], r.get("sample_idx"))] = r.get("reasoning_text")
    cuts = [json.loads(l) for l in open(a.cut_file)]
    cuts = [c for c in cuts if c.get("adaptive_truncated_text") and (c["id"], c.get("sample_idx")) in full]
    rnd = random.Random(a.seed)
    corr = [c for c in cuts if c["correct"]]
    inc = [c for c in cuts if not c["correct"]]
    rnd.shuffle(corr)
    rnd.shuffle(inc)
    groups = {"correct": corr[:a.n_correct], "incorrect": inc[:a.n_incorrect]}
    print(f"Scoring {len(groups['correct'])} correct and {len(groups['incorrect'])} incorrect traces "
          f"(available: {len(corr)} / {len(inc)})")

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(STUDENT_MODEL)
    model = AutoModelForCausalLM.from_pretrained(STUDENT_MODEL, dtype=torch.bfloat16).to(dev).eval()
    m500 = load_math500_lookup()

    skipped = {"correct": 0, "incorrect": 0}
    res = {"cut": {"correct": [], "incorrect": []}, "full": {"correct": [], "incorrect": []}}
    for label, rows in groups.items():
        for i, c in enumerate(rows):
            prob = get_problem_text(m500, c["id"])
            if not prob:
                continue
            try:
                lc = score_gold_answer_logprob(model, tok, prob, c["adaptive_truncated_text"], c["answer"], dev)
                lf = score_gold_answer_logprob(model, tok, prob, full[(c["id"], c.get("sample_idx"))], c["answer"], dev)
            except RuntimeError as e:          # CUDA out of memory on a very long trace: skip the PAIR
                if "out of memory" not in str(e).lower():
                    raise
                skipped[label] += 1
                torch.cuda.empty_cache()
                continue
            if lc is not None and lf is not None:
                res["cut"][label].append(lc)
                res["full"][label].append(lf)
            if (i + 1) % 20 == 0:
                print(f"  [{label}] {i + 1}/{len(rows)}")

    print(f"\nSkipped (out of memory): correct={skipped['correct']} incorrect={skipped['incorrect']}")
    print("\nVariant | mean logp correct | mean logp incorrect | separation")
    out = {}
    for v in ["cut", "full"]:
        cm, im = mean(res[v]["correct"]), mean(res[v]["incorrect"])
        out[v] = {"correct": cm, "incorrect": im, "separation": cm - im,
                  "n_correct": len(res[v]["correct"]), "n_incorrect": len(res[v]["incorrect"])}
        print(f"{v:<7} | {cm:.3f} | {im:.3f} | {cm - im:.3f}")
    sc, sf = out["cut"]["separation"], out["full"]["separation"]
    print(f"\nRetention (cut separation / full separation): {100 * sc / sf:.1f}%")
    # bootstrap the retention ratio by resampling traces within each group
    rnd = random.Random(1)
    ratios = []
    nc, ni = len(res["cut"]["correct"]), len(res["cut"]["incorrect"])
    for _ in range(2000):
        ci = [rnd.randrange(nc) for _ in range(nc)]
        ii = [rnd.randrange(ni) for _ in range(ni)]
        s_cut = mean([res["cut"]["correct"][k] for k in ci]) - mean([res["cut"]["incorrect"][k] for k in ii])
        s_full = mean([res["full"]["correct"][k] for k in ci]) - mean([res["full"]["incorrect"][k] for k in ii])
        if s_full > 0:
            ratios.append(s_cut / s_full)
    ratios.sort()
    if ratios:
        print(f"Bootstrap 95% interval for retention: {100 * ratios[int(0.025 * len(ratios))]:.0f}% to "
              f"{100 * ratios[int(0.975 * len(ratios))]:.0f}%")
    json.dump(out, open(a.out, "w"), indent=2)
    print(f"Saved {a.out}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--cut-file", required=True)
    p.add_argument("--trace-file", required=True)
    p.add_argument("--n-correct", type=int, default=60)
    p.add_argument("--n-incorrect", type=int, default=60)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default="score_cut_v2_summary.json")
    main(p.parse_args())
