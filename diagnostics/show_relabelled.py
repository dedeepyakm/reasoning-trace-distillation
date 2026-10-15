"""Lists every trace whose correct/incorrect label CHANGED when recomputed by common/answer_matching.
Use it to eyeball that each flip is right. Read: stored label -> new label, gold vs the model's answer.

Usage (repo root):
    python diagnostics/show_relabelled.py --trace-file data/traces_reasoning_teacher_k5_v2_clean.jsonl --cut-file data/cut_reasoning_v5.jsonl
"""
import argparse
import json


def main(trace_file, cut_file, tail):
    new = {}
    for line in open(cut_file):
        r = json.loads(line)
        new[(r["id"], r.get("sample_idx"))] = r["correct"]
    n_up = n_down = 0
    for line in open(trace_file):
        r = json.loads(line)
        key = (r["id"], r.get("sample_idx"))
        if key not in new or bool(r.get("correct")) == new[key]:
            continue
        direction = "INCORRECT -> CORRECT" if new[key] else "CORRECT -> INCORRECT"
        n_up += new[key]
        n_down += not new[key]
        print("=" * 78)
        print(f"{r['id']} sample={r.get('sample_idx')}  {direction}")
        print(f"  gold      : {r['answer']}")
        print(f"  model said: {r.get('predicted_answer', r.get('pred'))}")
        print(f"  trace ends: ...{r['reasoning_text'][-tail:]!r}")
    print("=" * 78)
    print(f"Flipped to correct: {n_up}   Flipped to incorrect: {n_down}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--trace-file", required=True)
    p.add_argument("--cut-file", required=True)
    p.add_argument("--tail", type=int, default=160)
    a = p.parse_args()
    main(a.trace_file, a.cut_file, a.tail)
