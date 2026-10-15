"""
Score the adaptive-cut traces (from adaptive_cut_analysis.py) with the SAME
teacher-forced log-prob method used in partition_heuristic.py, so the result
is directly comparable to your prefix/middle/suffix separation table.

For each trace: scores the student's confidence in the gold answer given
(a) the adaptively-cut reasoning text, and, for reference, (b) the full
original reasoning text -- both using the identical prompt template and
scoring function as partition_heuristic.py.

Requires:
    --cut-file    adaptive_cut_traces.jsonl  (from adaptive_cut_analysis.py)
    --trace-file  traces_teacher_k5.jsonl    (original, for full-trace reference
                                               and to confirm ids line up)

Usage:
    python score_adaptive_cut.py --cut-file adaptive_cut_traces.jsonl --trace-file traces_teacher_k5.jsonl --n-correct 40 --n-incorrect 40
"""

import argparse
import json
import re
from collections import defaultdict

import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

STUDENT_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"
PROMPT_TEMPLATE = (
    "Solve the following math problem step by step.\n\nProblem: {problem}\n\n"
    "{prefix_block}Answer: \\boxed{{"
)


def load_cut_rows(cut_file, n_correct, n_incorrect):
    seen_ids = set()
    correct_rows, incorrect_rows = [], []
    with open(cut_file) as f:
        for line in f:
            row = json.loads(line)
            pid = row.get("id")
            if pid in seen_ids:
                continue
            if row.get("adaptive_truncated_text") is None or row.get("answer") is None:
                continue
            if row["correct"] and len(correct_rows) < n_correct:
                correct_rows.append(row)
                seen_ids.add(pid)
            elif (not row["correct"]) and len(incorrect_rows) < n_incorrect:
                incorrect_rows.append(row)
                seen_ids.add(pid)
            if len(correct_rows) >= n_correct and len(incorrect_rows) >= n_incorrect:
                break
    return correct_rows, incorrect_rows


def load_full_reasoning_lookup(trace_file):
    """id -> full original reasoning_text, for the reference (uncut) score."""
    lookup = {}
    with open(trace_file) as f:
        for line in f:
            row = json.loads(line)
            pid = row.get("id")
            if pid is not None and pid not in lookup and row.get("reasoning_text") is not None:
                lookup[pid] = row["reasoning_text"]
    return lookup


@torch.no_grad()
def score_gold_answer_logprob(model, tokenizer, problem, prefix_text, gold_answer, device):
    """Identical to partition_heuristic.py's function -- teacher-forced mean
    log-prob per answer token, given problem + a reasoning prefix."""
    prefix_block = f"Partial reasoning so far: {prefix_text}\n\n" if prefix_text else ""
    prompt = PROMPT_TEMPLATE.format(problem=problem, prefix_block=prefix_block)
    target = f"{gold_answer}}}"

    prompt_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)
    full_ids = tokenizer(prompt + target, return_tensors="pt").input_ids.to(device)

    if full_ids.shape[1] <= prompt_ids.shape[1]:
        return None

    outputs = model(full_ids)
    logits = outputs.logits[0]
    log_probs = torch.log_softmax(logits, dim=-1)

    start = prompt_ids.shape[1]
    target_token_ids = full_ids[0, start:]
    relevant_log_probs = log_probs[start - 1 : full_ids.shape[1] - 1]
    token_logps = relevant_log_probs.gather(1, target_token_ids.unsqueeze(1)).squeeze(1)
    return token_logps.mean().item()


def load_math500_lookup():
    return load_dataset("HuggingFaceH4/MATH-500", split="test")


def get_problem_text(dataset, row_id):
    m = re.search(r"(\d+)$", row_id)
    if not m:
        return None
    idx = int(m.group(1))
    if idx < 0 or idx >= len(dataset):
        return None
    return dataset[idx]["problem"]


def main(cut_file, trace_file, n_correct, n_incorrect, out_path):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading student model on {device}...")
    tokenizer = AutoTokenizer.from_pretrained(STUDENT_MODEL)
    model = AutoModelForCausalLM.from_pretrained(STUDENT_MODEL, torch_dtype=torch.bfloat16).to(device)
    model.eval()

    print("Loading MATH-500 dataset for problem text lookup...")
    math500 = load_math500_lookup()

    print(f"Loading adaptive-cut traces from {cut_file}...")
    correct_rows, incorrect_rows = load_cut_rows(cut_file, n_correct, n_incorrect)
    print(f"Loaded {len(correct_rows)} correct, {len(incorrect_rows)} incorrect cut traces")

    print(f"Loading full original reasoning from {trace_file} for reference scores...")
    full_lookup = load_full_reasoning_lookup(trace_file)

    results = {
        "adaptive_cut": defaultdict(list),
        "full_original": defaultdict(list),
    }

    def run_group(rows, label):
        for i, row in enumerate(rows):
            problem = get_problem_text(math500, row["id"])
            if not problem:
                continue
            gold = row["answer"]

            cut_logp = score_gold_answer_logprob(model, tokenizer, problem, row["adaptive_truncated_text"], gold, device)
            if cut_logp is not None:
                results["adaptive_cut"][label].append(cut_logp)

            full_text = full_lookup.get(row["id"])
            if full_text is not None:
                full_logp = score_gold_answer_logprob(model, tokenizer, problem, full_text, gold, device)
                if full_logp is not None:
                    results["full_original"][label].append(full_logp)

            if (i + 1) % 10 == 0:
                print(f"  [{label}] {i + 1}/{len(rows)} scored")

    run_group(correct_rows, "correct")
    run_group(incorrect_rows, "incorrect")

    print("\nVariant         | mean logp (correct) | mean logp (incorrect) | separation")
    summary = {}
    for variant in ["adaptive_cut", "full_original"]:
        c_vals = results[variant]["correct"]
        i_vals = results[variant]["incorrect"]
        c_mean = sum(c_vals) / len(c_vals) if c_vals else None
        i_mean = sum(i_vals) / len(i_vals) if i_vals else None
        sep = (c_mean - i_mean) if (c_mean is not None and i_mean is not None) else None
        print(f"{variant:<16} | {c_mean if c_mean is not None else 'N/A':>20} | "
              f"{i_mean if i_mean is not None else 'N/A':>22} | {sep if sep is not None else 'N/A'}")
        summary[variant] = {"correct_mean_logp": c_mean, "incorrect_mean_logp": i_mean, "separation": sep}

    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSaved summary to {out_path}")
    print(
        "\nCompare 'adaptive_cut' separation directly to your prefix/middle/suffix "
        "table from partition_heuristic.py -- it tells you whether cutting each "
        "trace right at its own first explicit-answer point keeps as much (or more) "
        "signal as the fixed 50%/75% suffix cuts did, at a smaller average token cost."
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cut-file", required=True, help="adaptive_cut_traces.jsonl from adaptive_cut_analysis.py")
    parser.add_argument("--trace-file", required=True, help="original traces_teacher_k5.jsonl, for full-trace reference score")
    parser.add_argument("--n-correct", type=int, default=40)
    parser.add_argument("--n-incorrect", type=int, default=40)
    parser.add_argument("--out", default="adaptive_cut_score_summary.json")
    args = parser.parse_args()
    main(args.cut_file, args.trace_file, args.n_correct, args.n_incorrect, args.out)
