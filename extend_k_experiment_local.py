"""
Extend k-value experiment -- LOCAL LAPTOP VERSION (Windows, RTX 5070 8GB VRAM)

Uses plain HF `transformers` generation instead of vLLM, since vLLM does not
run natively on Windows (Linux/WSL2 only).

IMPORTANT: this script is sized for the STUDENT model (Qwen2.5-1.5B-Instruct)
only. The teacher (Qwen2.5-7B-Instruct) needs ~14GB VRAM at bf16, which will
not fit in 8GB -- run the teacher on Phoenix instead. If you really want to
attempt the teacher locally, see the NOTE near TEACHER_LOCAL_ATTEMPT below,
but expect this to be slow and possibly OOM.

Usage:
    python extend_k_experiment_local.py --role student

Generates n=8 samples per problem (one at a time, to stay within VRAM),
then computes pass@1..pass@8 from that pool using the unbiased estimator.
Saves incrementally so you can stop/resume.
"""

import argparse
import gc
import json
from pathlib import Path

import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

MODEL_PATHS = {
    "student": "Qwen/Qwen2.5-1.5B-Instruct",
    # NOTE / TEACHER_LOCAL_ATTEMPT: only uncomment if you want to try it locally.
    # Expect this to be slow, and it may OOM even at 4-bit. Prefer Phoenix.
    # "teacher": "Qwen/Qwen2.5-7B-Instruct",
}

N_SAMPLES = 8
MAX_K = 8
TEMPERATURE = 0.8
TOP_P = 0.95
MAX_NEW_TOKENS = 1024   # reduced from 2048 for local speed/VRAM; bump back up on Phoenix

OUTPUT_DIR = Path("./k_extension_results")
OUTPUT_DIR.mkdir(exist_ok=True)

PROMPT_TEMPLATE = (
    "Solve the following math problem step by step. "
    "Put your final answer in \\boxed{{}}.\n\nProblem: {problem}\n\nSolution:"
)

# ---------------------------------------------------------------------------
# Answer extraction and checking (same as Phoenix version)
# ---------------------------------------------------------------------------

def extract_boxed_answer(text: str) -> str | None:
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


def pass_at_k(n: int, c: int, k: int) -> float:
    if n - c < k:
        return 1.0
    from math import comb
    return 1.0 - comb(n - c, k) / comb(n, k)


# ---------------------------------------------------------------------------
# Resume support
# ---------------------------------------------------------------------------

def load_existing(role: str) -> dict:
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


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(role: str):
    assert role in MODEL_PATHS, f"role must be one of {list(MODEL_PATHS)}"
    model_path = MODEL_PATHS[role]
    print(f"[{role}] loading model: {model_path}")
    print(f"[{role}] CUDA available: {torch.cuda.is_available()}")

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        device_map="cuda",
    )
    model.eval()

    dataset = load_dataset("HuggingFaceH4/MATH-500", split="test")
    print(f"[{role}] loaded {len(dataset)} problems from MATH-500")

    already_done = load_existing(role)
    print(f"[{role}] {len(already_done)} problems already sampled, resuming rest")

    todo_indices = [i for i in range(len(dataset)) if str(i) not in already_done]
    print(f"[{role}] {len(todo_indices)} problems remaining")

    for count, idx in enumerate(todo_indices):
        problem = dataset[idx]["problem"]
        gold = dataset[idx]["answer"]
        prompt = PROMPT_TEMPLATE.format(problem=problem)

        messages = [{"role": "user", "content": prompt}]
        encoding = tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, return_tensors="pt", return_dict=True
        ).to(model.device)
        input_ids = encoding["input_ids"]
        attention_mask = encoding.get("attention_mask")

        completions = []
        for sample_i in range(N_SAMPLES):
            with torch.no_grad():
                output = model.generate(
                    input_ids,
                    attention_mask=attention_mask,
                    max_new_tokens=MAX_NEW_TOKENS,
                    do_sample=True,
                    temperature=TEMPERATURE,
                    top_p=TOP_P,
                    pad_token_id=tokenizer.eos_token_id,
                )
                text = tokenizer.decode(
                output[0][input_ids.shape[1]:], skip_special_tokens=True
            )
            completions.append(text)

        correctness = [is_correct(c, gold) for c in completions]
        row = {
            "problem_id": str(idx),
            "difficulty": dataset[idx].get("level", None),
            "gold_answer": gold,
            "n_samples": len(completions),
            "n_correct": sum(correctness),
            "correctness": correctness,
        }
        save_row(role, row)
        already_done[str(idx)] = row

        if count % 5 == 0:
            gc.collect()
            torch.cuda.empty_cache()
            mem_gb = torch.cuda.memory_allocated() / 1e9
            print(f"[{role}] {count+1}/{len(todo_indices)} done "
                  f"(problem {idx}: {sum(correctness)}/{N_SAMPLES} correct, "
                  f"GPU mem: {mem_gb:.1f} GB)")

    # ---- compute pass@k for k = 1..MAX_K ----
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
    parser.add_argument("--role", required=True, choices=list(MODEL_PATHS.keys()))
    args = parser.parse_args()
    main(args.role)
