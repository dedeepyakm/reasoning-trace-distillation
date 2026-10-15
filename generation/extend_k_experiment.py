"""
Extend k-value experiments for reasoning trace distillation project.

Generates n=8 samples per problem for a given model on MATH-500, then computes
pass@1 through pass@8 using the unbiased estimator from Chen et al. 2021
(the Codex paper). This avoids regenerating separately for each k -- you
generate once at n=8 and derive every smaller k from that same pool.

Usage:
    python extend_k_experiment.py --role teacher
    python extend_k_experiment.py --role student

Run both (teacher and student) as separate jobs if you have GPU capacity,
or sequentially if not. Results are saved incrementally per problem, so if
the job dies partway through you can rerun and it will skip completed ones.
"""

import argparse
import json
import os
import re
from pathlib import Path

from datasets import load_dataset
from vllm import LLM, SamplingParams

# ---------------------------------------------------------------------------
# Config -- edit these to match your setup
# ---------------------------------------------------------------------------

MODEL_PATHS = {
    "teacher": "Qwen/Qwen2.5-7B-Instruct",
    "student": "Qwen/Qwen2.5-1.5B-Instruct",
}

N_SAMPLES = 8          # generate this many completions per problem
MAX_K = 8               # compute pass@k up to this value
TEMPERATURE = 0.8        # standard for pass@k sampling (need diversity across samples)
TOP_P = 0.95
MAX_NEW_TOKENS = 2048    # bump to 4096+ later for the full-precision Phoenix rerun

OUTPUT_DIR = Path("./k_extension_results")
OUTPUT_DIR.mkdir(exist_ok=True)

PROMPT_TEMPLATE = (
    "Solve the following math problem step by step. "
    "Put your final answer in \\boxed{{}}.\n\nProblem: {problem}\n\nSolution:"
)

# ---------------------------------------------------------------------------
# Answer extraction and checking
# ---------------------------------------------------------------------------

def extract_boxed_answer(text: str) -> str | None:
    """Extract the content of the last \\boxed{...} in the text, handling nested braces."""
    idx = text.rfind("\\boxed{")
    if idx == -1:
        return None
    start = idx + len("\\boxed{")
    depth = 1
    i = start
    while i < len(text) and depth > 0:
        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            depth -= 1
        i += 1
    if depth != 0:
        return None
    return text[start : i - 1].strip()


def normalize_answer(ans: str) -> str:
    """Light normalization -- strip whitespace, dollar signs, trailing punctuation.
    NOTE: swap this out for your existing checker if you already have one that
    handles fractions/equivalent forms more rigorously (e.g. via sympy)."""
    if ans is None:
        return ""
    ans = ans.strip().replace(" ", "")
    ans = ans.strip("$")
    ans = ans.rstrip(".")
    return ans


def is_correct(generated_text: str, gold_answer: str) -> bool:
    pred = extract_boxed_answer(generated_text)
    if pred is None:
        return False
    return normalize_answer(pred) == normalize_answer(gold_answer)


# ---------------------------------------------------------------------------
# Unbiased pass@k estimator (Chen et al. 2021, "Evaluating Large Language
# Models Trained on Code")
# ---------------------------------------------------------------------------

def pass_at_k(n: int, c: int, k: int) -> float:
    """
    n: total samples generated for this problem
    c: number of those that were correct
    k: the k in pass@k
    """
    if n - c < k:
        return 1.0
    from math import comb
    return 1.0 - comb(n - c, k) / comb(n, k)


# ---------------------------------------------------------------------------
# Main generation + evaluation loop
# ---------------------------------------------------------------------------

def load_existing(role: str) -> dict:
    """Resume support: load any problems already fully sampled in a prior run."""
    path = OUTPUT_DIR / f"{role}_raw_samples.jsonl"
    done = {}
    if path.exists():
        with open(path) as f:
            for line in f:
                row = json.loads(line)
                done[row["problem_id"]] = row
    return done


def save_row(role: str, row: dict):
    path = OUTPUT_DIR / f"{role}_raw_samples.jsonl"
    with open(path, "a") as f:
        f.write(json.dumps(row) + "\n")


def main(role: str):
    assert role in MODEL_PATHS, f"role must be 'teacher' or 'student', got {role}"
    model_path = MODEL_PATHS[role]
    print(f"[{role}] loading model: {model_path}")

    llm = LLM(model=model_path, dtype="bfloat16", gpu_memory_utilization=0.9)
    sampling_params = SamplingParams(
        n=N_SAMPLES,
        temperature=TEMPERATURE,
        top_p=TOP_P,
        max_tokens=MAX_NEW_TOKENS,
    )

    dataset = load_dataset("HuggingFaceH4/MATH-500", split="test")
    print(f"[{role}] loaded {len(dataset)} problems from MATH-500")

    already_done = load_existing(role)
    print(f"[{role}] {len(already_done)} problems already sampled, resuming rest")

    todo_indices = [i for i in range(len(dataset)) if str(i) not in already_done]
    if not todo_indices:
        print(f"[{role}] nothing left to generate, skipping to pass@k computation")
    else:
        prompts = [
            PROMPT_TEMPLATE.format(problem=dataset[i]["problem"]) for i in todo_indices
        ]
        print(f"[{role}] generating {N_SAMPLES} samples each for {len(prompts)} problems...")
        outputs = llm.generate(prompts, sampling_params)

        for idx, output in zip(todo_indices, outputs):
            gold = dataset[idx]["answer"]
            completions = [o.text for o in output.outputs]
            correctness = [is_correct(c, gold) for c in completions]
            row = {
                "problem_id": str(idx),
                "difficulty": dataset[idx].get("level", None),
                "gold_answer": gold,
                "n_samples": len(completions),
                "n_correct": sum(correctness),
                "correctness": correctness,  # per-sample bool list, kept for auditing
            }
            save_row(role, row)
            already_done[str(idx)] = row

    # ---- compute pass@k for k = 1..MAX_K from the n=8 pool ----
    results_by_k = {k: [] for k in range(1, MAX_K + 1)}
    for row in already_done.values():
        n = row["n_samples"]
        c = row["n_correct"]
        for k in range(1, MAX_K + 1):
            if k > n:
                continue
            results_by_k[k].append(pass_at_k(n, c, k))

    summary = {
        f"pass@{k}": (sum(vals) / len(vals) if vals else None)
        for k, vals in results_by_k.items()
    }
    summary_path = OUTPUT_DIR / f"{role}_passk_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"[{role}] pass@k summary:")
    for k, v in summary.items():
        print(f"  {k}: {v:.4f}" if v is not None else f"  {k}: N/A")
    print(f"[{role}] saved to {summary_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--role", required=True, choices=["teacher", "student"])
    args = parser.parse_args()
    main(args.role)
