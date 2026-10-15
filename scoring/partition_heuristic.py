"""
Partition heuristic experiment -- graded prefix lengths.

Builds on the EntropyLong confound finding: instead of one blunt 50/50 cut,
test several prefix lengths of each teacher reasoning trace, and measure how
much the student model's confidence in the CORRECT answer improves as more
of the trace is revealed -- with the final answer text always excluded, so
there's no read-off shortcut.

If confidence separation (correct-trace prefixes vs incorrect-trace prefixes)
emerges early (e.g. by 25%), that suggests the useful reasoning is front-loaded.
If it only emerges late (75-100%), the useful part is likely near the end.

Uses your EXISTING trace file -- no new teacher generation needed.

Usage:
    python partition_heuristic.py --trace-file traces_Qwen2-5-7B-Instruct_math500.jsonl --n 60
"""

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path

import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

STUDENT_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"
PREFIX_FRACTIONS = [0.0, 0.10, 0.25, 0.50, 0.75, 1.0]
PROMPT_TEMPLATE = (
    "Solve the following math problem step by step.\n\nProblem: {problem}\n\n"
    "{prefix_block}Answer: \\boxed{{"
)


def load_traces(path, n_correct, n_incorrect):
    """Load up to n_correct correct-teacher-trace rows and n_incorrect incorrect
    ones, one row per problem (first sample_idx seen for each id)."""
    seen_ids = set()
    correct_rows, incorrect_rows = [], []
    with open(path) as f:
        for line in f:
            row = json.loads(line)
            pid = row["id"]
            if pid in seen_ids:
                continue
            if row.get("reasoning_text") is None or row.get("answer") is None:
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


def strip_last_sentences(text, n_sentences):
    """Remove the last n_sentences complete sentences from text (rough split on
    '.', '?', '!'). Used to test whether the suffix advantage is driven by a
    concluding, answer-revealing sentence rather than genuine reasoning."""
    if n_sentences <= 0:
        return text
    # Split keeping delimiters, then drop the last n non-empty chunks
    parts = re.split(r'(?<=[.?!])\s+', text.strip())
    parts = [p for p in parts if p]
    if len(parts) <= n_sentences:
        return ""
    return " ".join(parts[:-n_sentences])


def truncate_by_fraction(text, fraction, strategy="prefix", strip_last_n=0):
    """Truncate reasoning text to a `fraction` of its whitespace tokens, using
    one of three strategies:
      - prefix: keep the FIRST fraction (beginning-anchored)
      - suffix: keep the LAST fraction (end-anchored)
      - middle: keep a CENTERED window of that fraction (drop both ends)
    If strip_last_n > 0, the last N sentences are removed from the FULL text
    BEFORE truncation is applied -- used to test whether suffix's advantage
    survives removing a likely answer-revealing concluding sentence.
    """
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
    """Teacher-forced scoring: how likely does the student think the gold
    answer is, given the problem and a (possibly truncated, always answer-free)
    reasoning prefix. Returns mean log-prob per answer token (higher = more
    confident/correct-leaning)."""
    prefix_block = f"Partial reasoning so far: {prefix_text}\n\n" if prefix_text else ""
    prompt = PROMPT_TEMPLATE.format(problem=problem, prefix_block=prefix_block)
    target = f"{gold_answer}}}"  # closes the \boxed{ opened in the prompt template

    prompt_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)
    full_ids = tokenizer(prompt + target, return_tensors="pt").input_ids.to(device)

    if full_ids.shape[1] <= prompt_ids.shape[1]:
        return None  # target tokenized to nothing new, skip

    outputs = model(full_ids)
    logits = outputs.logits[0]  # [seq_len, vocab]
    log_probs = torch.log_softmax(logits, dim=-1)

    start = prompt_ids.shape[1]
    target_token_ids = full_ids[0, start:]
    # logits at position i predict token i+1, so shift accordingly
    relevant_log_probs = log_probs[start - 1 : full_ids.shape[1] - 1]
    token_logps = relevant_log_probs.gather(1, target_token_ids.unsqueeze(1)).squeeze(1)
    return token_logps.mean().item()


def load_math500_lookup():
    """id like 'math500-37' maps to dataset index 37."""
    dataset = load_dataset("HuggingFaceH4/MATH-500", split="test")
    return dataset


def get_problem_text(dataset, row_id):
    m = re.search(r"(\d+)$", row_id)
    if not m:
        return None
    idx = int(m.group(1))
    if idx < 0 or idx >= len(dataset):
        return None
    return dataset[idx]["problem"]


def main(trace_file, n_correct, n_incorrect, out_path, strategy, fractions, strip_last_n):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading student model on {device}...")
    tokenizer = AutoTokenizer.from_pretrained(STUDENT_MODEL)
    model = AutoModelForCausalLM.from_pretrained(STUDENT_MODEL, torch_dtype=torch.bfloat16).to(device)
    model.eval()

    print("Loading MATH-500 dataset for problem text lookup...")
    math500 = load_math500_lookup()

    print(f"Loading traces from {trace_file}...")
    correct_rows, incorrect_rows = load_traces(trace_file, n_correct, n_incorrect)
    print(f"Loaded {len(correct_rows)} correct-trace problems, {len(incorrect_rows)} incorrect-trace problems")
    print(f"Strategy: {strategy}, fractions: {fractions}, strip_last_n: {strip_last_n}")

    results = defaultdict(lambda: {"correct": [], "incorrect": []})

    def run_group(rows, label):
        for i, row in enumerate(rows):
            problem = get_problem_text(math500, row["id"])
            if not problem:
                continue
            gold = row["answer"]
            reasoning = row["reasoning_text"]
            for frac in fractions:
                prefix = truncate_by_fraction(reasoning, frac, strategy=strategy, strip_last_n=strip_last_n)
                logp = score_gold_answer_logprob(model, tokenizer, problem, prefix, gold, device)
                if logp is not None:
                    results[frac][label].append(logp)
            if (i + 1) % 10 == 0:
                print(f"  [{label}] {i + 1}/{len(rows)} problems scored")

    run_group(correct_rows, "correct")
    run_group(incorrect_rows, "incorrect")

    print("\nPrefix fraction | mean logp (correct-trace) | mean logp (incorrect-trace) | separation")
    summary = {"strategy": strategy, "strip_last_n": strip_last_n, "by_fraction": {}}
    for frac in fractions:
        c_vals = results[frac]["correct"]
        i_vals = results[frac]["incorrect"]
        c_mean = sum(c_vals) / len(c_vals) if c_vals else None
        i_mean = sum(i_vals) / len(i_vals) if i_vals else None
        sep = (c_mean - i_mean) if (c_mean is not None and i_mean is not None) else None
        print(f"{frac:>15.2f} | {c_mean if c_mean is not None else 'N/A':>25} | "
              f"{i_mean if i_mean is not None else 'N/A':>27} | {sep if sep is not None else 'N/A'}")
        summary["by_fraction"][str(frac)] = {"correct_mean_logp": c_mean, "incorrect_mean_logp": i_mean, "separation": sep}

    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSaved summary to {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--trace-file", required=True, help="Path to an existing teacher traces .jsonl file")
    parser.add_argument("--n-correct", type=int, default=40)
    parser.add_argument("--n-incorrect", type=int, default=40)
    parser.add_argument("--out", default="partition_heuristic_summary.json")
    parser.add_argument("--strategy", choices=["prefix", "suffix", "middle"], default="prefix",
                         help="Which part of the trace to keep: prefix (start), suffix (end), or middle (centered window)")
    parser.add_argument("--fractions", type=float, nargs="+", default=[0.0, 0.10, 0.25, 0.50, 0.75, 1.0])
    parser.add_argument("--strip-last-n", type=int, default=0,
                         help="Remove the last N sentences from the trace BEFORE truncation, to test for concluding-sentence leakage")
    args = parser.parse_args()
    main(args.trace_file, args.n_correct, args.n_incorrect, args.out, args.strategy, args.fractions, args.strip_last_n)
