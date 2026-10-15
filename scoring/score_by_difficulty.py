"""
Signal-retention scoring, split by MATH-500 difficulty tier (Easy / Medium / Hard).

This does two things in one run:

1. Closes last week's scale mismatch: prefix / middle / suffix-stripped /
   adaptive-cut / full-trace are all scored on the SAME 100-problem combined
   dataset (previously the fixed-fraction baselines were still from the
   original 50-problem run).

2. Answers the "does this work on complex questions too" question directly:
   every number is reported separately for Easy (MATH level 1-2) and Hard
   (level 4-5) problems, so we can see whether adaptive cutting still beats
   the fixed-fraction baselines when the problems are actually difficult --
   not just in an average that's mostly easy problems.

Requires:
    --cut-file    adaptive_cut_traces_by_difficulty.jsonl  (from difficulty_stratified_analysis.py)
    --trace-file  traces_teacher_k5_combined.jsonl         (the original combined trace file)

Usage:
    python score_by_difficulty.py --cut-file adaptive_cut_traces_by_difficulty.jsonl --trace-file traces_teacher_k5_combined.jsonl --n-per-tier 25
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
FRACTIONS_TO_SCORE = ["prefix75", "middle75", "suffix75_stripped", "adaptive_cut", "full_original"]


def strip_last_sentences(text, n_sentences):
    if n_sentences <= 0:
        return text
    parts = re.split(r'(?<=[.?!])\s+', text.strip())
    parts = [p for p in parts if p]
    if len(parts) <= n_sentences:
        return ""
    return " ".join(parts[:-n_sentences])


def truncate_by_fraction(text, fraction, strategy="prefix", strip_last_n=0):
    if strip_last_n > 0:
        text = strip_last_sentences(text, strip_last_n)
    tokens = text.split()
    if fraction <= 0:
        return ""
    if fraction >= 1.0:
        return text
    n = len(tokens)
    keep = max(1, int(n * fraction))
    if strategy == "prefix":
        chosen = tokens[:keep]
    elif strategy == "suffix":
        chosen = tokens[n - keep:]
    elif strategy == "middle":
        start = max(0, (n - keep) // 2)
        chosen = tokens[start:start + keep]
    else:
        raise ValueError(f"unknown strategy {strategy}")
    return " ".join(chosen)


@torch.no_grad()
def score_gold_answer_logprob(model, tokenizer, problem, prefix_text, gold_answer, device):
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


def load_rows(cut_file, n_per_tier):
    by_tier = defaultdict(lambda: {"correct": [], "incorrect": []})
    seen_ids = set()
    with open(cut_file) as f:
        for line in f:
            row = json.loads(line)
            pid = row.get("id")
            tier = row.get("difficulty_tier")
            if pid in seen_ids or tier is None or row.get("adaptive_truncated_text") is None:
                continue
            label = "correct" if row["correct"] else "incorrect"
            if len(by_tier[tier][label]) < n_per_tier:
                by_tier[tier][label].append(row)
                seen_ids.add(pid)
    return by_tier


def load_full_reasoning_lookup(trace_file):
    lookup = {}
    with open(trace_file) as f:
        for line in f:
            row = json.loads(line)
            pid = row.get("id")
            if pid is not None and pid not in lookup and row.get("reasoning_text") is not None:
                lookup[pid] = row["reasoning_text"]
    return lookup


def load_math500():
    return load_dataset("HuggingFaceH4/MATH-500", split="test")


def get_problem_text(dataset, row_id):
    m = re.search(r"(\d+)$", row_id)
    if not m:
        return None
    idx = int(m.group(1))
    if idx < 0 or idx >= len(dataset):
        return None
    return dataset[idx]["problem"]


def main(cut_file, trace_file, n_per_tier, out_path):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading student model on {device}...")
    tokenizer = AutoTokenizer.from_pretrained(STUDENT_MODEL)
    model = AutoModelForCausalLM.from_pretrained(STUDENT_MODEL, torch_dtype=torch.bfloat16).to(device)
    model.eval()

    print("Loading MATH-500 dataset...")
    math500 = load_math500()

    print(f"Loading tagged cut traces from {cut_file}...")
    by_tier = load_rows(cut_file, n_per_tier)
    for tier, groups in by_tier.items():
        print(f"  {tier}: {len(groups['correct'])} correct, {len(groups['incorrect'])} incorrect")

    print(f"Loading full original reasoning from {trace_file}...")
    full_lookup = load_full_reasoning_lookup(trace_file)

    results = defaultdict(lambda: defaultdict(list))

    def score_one(row, variant, problem):
        gold = row["answer"]
        full_text = full_lookup.get(row["id"])
        if full_text is None:
            return None
        if variant == "adaptive_cut":
            prefix = row["adaptive_truncated_text"]
        elif variant == "full_original":
            prefix = full_text
        elif variant == "prefix75":
            prefix = truncate_by_fraction(full_text, 0.75, strategy="prefix")
        elif variant == "middle75":
            prefix = truncate_by_fraction(full_text, 0.75, strategy="middle")
        elif variant == "suffix75_stripped":
            prefix = truncate_by_fraction(full_text, 0.75, strategy="suffix", strip_last_n=2)
        else:
            raise ValueError(variant)
        return score_gold_answer_logprob(model, tokenizer, problem, prefix, gold, device)

    for tier, groups in by_tier.items():
        for label in ["correct", "incorrect"]:
            rows = groups[label]
            for i, row in enumerate(rows):
                problem = get_problem_text(math500, row["id"])
                if not problem:
                    continue
                for variant in FRACTIONS_TO_SCORE:
                    logp = score_one(row, variant, problem)
                    if logp is not None:
                        results[(tier, variant)][label].append(logp)
                if (i + 1) % 10 == 0:
                    print(f"  [{tier}/{label}] {i + 1}/{len(rows)} scored")

    print("\nTier     | Variant             | mean logp (correct) | mean logp (incorrect) | separation")
    print("-" * 95)
    summary = {}
    for tier in ["Easy", "Medium", "Hard"]:
        for variant in FRACTIONS_TO_SCORE:
            c_vals = results[(tier, variant)]["correct"]
            i_vals = results[(tier, variant)]["incorrect"]
            c_mean = sum(c_vals) / len(c_vals) if c_vals else None
            i_mean = sum(i_vals) / len(i_vals) if i_vals else None
            sep = (c_mean - i_mean) if (c_mean is not None and i_mean is not None) else None
            c_str = f"{c_mean:.3f}" if c_mean is not None else "N/A"
            i_str = f"{i_mean:.3f}" if i_mean is not None else "N/A"
            sep_str = f"{sep:.3f}" if sep is not None else "N/A"
            print(f"{tier:<8} | {variant:<19} | {c_str:>20} | {i_str:>22} | {sep_str}")
            summary[f"{tier}/{variant}"] = {"correct_mean_logp": c_mean, "incorrect_mean_logp": i_mean, "separation": sep}

    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSaved summary to {out_path}")
    print(
        "\nRead this as: does 'adaptive_cut' still beat 'prefix75' / 'middle75' / "
        "'suffix75_stripped' within the Hard row, not just within Easy? That's the "
        "direct answer to whether this approach holds on complex questions."
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cut-file", required=True, help="adaptive_cut_traces_by_difficulty.jsonl")
    parser.add_argument("--trace-file", required=True, help="traces_teacher_k5_combined.jsonl")
    parser.add_argument("--n-per-tier", type=int, default=25, help="max correct/incorrect examples scored per tier")
    parser.add_argument("--out", default="score_by_difficulty_summary.json")
    args = parser.parse_args()
    main(args.cut_file, args.trace_file, args.n_per_tier, args.out)
