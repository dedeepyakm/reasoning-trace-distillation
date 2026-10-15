"""
Teacher k=8 extension -- REMOTE MACHINE VERSION (Ubuntu, RTX A2000 12GB)

Resumes from your existing traces_teacher_k5.jsonl (samples 0-4 per problem)
and generates only the missing samples (5, 6, 7) to reach k=8 -- much faster
than regenerating from scratch.

Uses 8-bit quantization via bitsandbytes -- a real step up from the nf4 (4-bit)
quantization used previously, meaningfully less lossy, and this 12GB card
handles it comfortably for the 7B teacher.

Usage:
    python3 teacher_k8_extension.py --existing traces_teacher_k5.jsonl --out traces_teacher_k8.jsonl

Matches the same schema as your existing trace files (id, dataset, level,
subject, answer, sample_idx, model, quantisation, sample_mode, sample_seed,
pred, correct, logp_r_sum, logp_r_mean, logp_y_sum, logp_y_mean, picsar_sum,
n_tokens_total, n_tokens_reasoning, n_tokens_answer, reasoning_text, answer_text)
"""

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path

import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

TEACHER_MODEL = "Qwen/Qwen2.5-7B-Instruct"
TARGET_K = 8
MAX_NEW_TOKENS = 1024
TEMPERATURE = 0.8
TOP_P = 0.95

PROMPT_TEMPLATE = (
    "Solve the following math problem step by step. "
    "Put your final answer in \\boxed{{}}.\n\nProblem: {problem}\n\nSolution:"
)


def extract_boxed_answer(text):
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
    return text[start:i - 1].strip()


def normalize_answer(ans):
    if ans is None:
        return ""
    return ans.strip().replace(" ", "").strip("$").rstrip(".")


def is_correct(generated_text, gold_answer):
    pred = extract_boxed_answer(generated_text)
    if pred is None:
        return False, None
    return normalize_answer(pred) == normalize_answer(gold_answer), pred


def load_existing_max_sample_idx(path):
    """Returns {problem_numeric_id: highest sample_idx already present}."""
    max_idx = defaultdict(lambda: -1)
    rows_by_id = defaultdict(list)
    if not Path(path).exists():
        return max_idx, rows_by_id
    with open(path) as f:
        for line in f:
            row = json.loads(line)
            m = re.search(r"(\d+)$", row["id"])
            if not m:
                continue
            pid = int(m.group(1))
            max_idx[pid] = max(max_idx[pid], row["sample_idx"])
            rows_by_id[pid].append(row)
    return max_idx, rows_by_id


def main(existing_path, out_path, existing_only):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading teacher model on {device} with 8-bit quantization...")

    bnb_config = BitsAndBytesConfig(load_in_8bit=True)
    tokenizer = AutoTokenizer.from_pretrained(TEACHER_MODEL)
    model = AutoModelForCausalLM.from_pretrained(
        TEACHER_MODEL,
        quantization_config=bnb_config,
        device_map="auto",
    )
    model.eval()

    print("Loading MATH-500 dataset...")
    dataset = load_dataset("HuggingFaceH4/MATH-500", split="test")

    print(f"Checking existing samples in {existing_path}...")
    max_idx, existing_rows = load_existing_max_sample_idx(existing_path)
    print(f"Found existing data for {len(max_idx)} problems")

    # copy existing rows forward into the output file first
    with open(out_path, "w") as out_f:
        for pid, rows in existing_rows.items():
            for row in rows:
                out_f.write(json.dumps(row) + "\n")

    total_needed = 0
    problem_ids_to_process = range(len(dataset)) if not existing_only else sorted(max_idx.keys())
    for i in problem_ids_to_process:
        need = TARGET_K - 1 - max_idx[i]
        if need > 0:
            total_needed += need
    print(f"Need to generate {total_needed} additional samples to reach k={TARGET_K} "
          f"for {len(list(problem_ids_to_process))} problems")

    generated_count = 0
    for i in problem_ids_to_process:
        start_idx = max_idx[i] + 1
        if start_idx >= TARGET_K:
            continue  # already at or past target for this problem

        problem = dataset[i]["problem"]
        gold = dataset[i]["answer"]
        level = dataset[i].get("level")
        subject = dataset[i].get("subject")
        prompt = PROMPT_TEMPLATE.format(problem=problem)

        for sample_idx in range(start_idx, TARGET_K):
            messages = [{"role": "user", "content": prompt}]
            encoding = tokenizer.apply_chat_template(
                messages, add_generation_prompt=True, return_tensors="pt", return_dict=True
            ).to(model.device)
            input_ids = encoding["input_ids"]
            attention_mask = encoding.get("attention_mask")

            with torch.no_grad():
                output = model.generate(
                    input_ids,
                    attention_mask=attention_mask,
                    max_new_tokens=MAX_NEW_TOKENS,
                    do_sample=True,
                    temperature=TEMPERATURE,
                    top_p=TOP_P,
                    pad_token_id=tokenizer.eos_token_id,
                    output_scores=True,
                    return_dict_in_generate=True,
                )

            gen_tokens = output.sequences[0][input_ids.shape[1]:]
            full_text = tokenizer.decode(gen_tokens, skip_special_tokens=True)
            correct, pred = is_correct(full_text, gold)

            # split reasoning vs answer text on the boxed marker, matching your existing schema
            boxed_pos = full_text.rfind("\\boxed{")
            if boxed_pos != -1:
                reasoning_text = full_text[:boxed_pos].strip()
                answer_text = full_text[boxed_pos:].strip()
            else:
                reasoning_text = full_text.strip()
                answer_text = ""

            row = {
                "id": f"math500-{i}",
                "dataset": "math500",
                "level": level,
                "subject": subject,
                "answer": gold,
                "sample_idx": sample_idx,
                "model": TEACHER_MODEL,
                "quantisation": "int8",
                "sample_mode": "stratified",
                "sample_seed": 0,
                "pred": pred,
                "correct": correct,
                "n_tokens_total": len(gen_tokens),
                "reasoning_text": reasoning_text,
                "answer_text": answer_text,
            }

            with open(out_path, "a") as f:
                f.write(json.dumps(row) + "\n")

            generated_count += 1
            if generated_count % 10 == 0:
                print(f"  {generated_count}/{total_needed} new samples generated "
                      f"(problem {i}, sample {sample_idx}: {'correct' if correct else 'incorrect'})")

    print(f"\nDone. Generated {generated_count} new samples.")
    print(f"Full k=8 dataset (old + new) saved to {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--existing", required=True, help="Path to existing k=5 trace file to resume from")
    parser.add_argument("--out", default="traces_teacher_k8.jsonl")
    parser.add_argument("--existing-only", action="store_true",
                         help="Only extend problems already present in the existing trace file, "
                              "skip generating fresh data for problems not yet covered")
    args = parser.parse_args()
    main(args.existing, args.out, args.existing_only)
