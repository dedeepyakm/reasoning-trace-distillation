"""
Local teacher trace generation (4-bit quantized) -- extends trace coverage
beyond the 50 MATH-500 problems already in traces_teacher_k5.jsonl, running
entirely on your laptop's 8GB GPU.

Uses 4-bit (nf4) quantization instead of the 8-bit setup from the remote
machine, since a 7B model in 8-bit (~7-8GB) is too tight alongside generation
activation memory on an 8GB card. 4-bit brings the weights down to ~4-5GB.

Generates k=5 samples per problem (matching your existing traces_teacher_k5.jsonl
schema: id, answer, reasoning_text, correct, sample_idx), for the NEXT batch of
MATH-500 problems not already covered by --existing-file. Resumable: writes
each problem's samples to the output file as soon as they're done, and skips
problems already present in --out if you rerun it.

Usage:
    python teacher_trace_generation_local.py --existing-file traces_teacher_k5.jsonl --out traces_teacher_k5_extended.jsonl --num-new-problems 50 --k 5
"""

import argparse
import json
import os
import re

import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

TEACHER_MODEL = "Qwen/Qwen2.5-7B-Instruct"

PROMPT_TEMPLATE = (
    "Solve the following math problem step by step. "
    "Put your final answer within \\boxed{{}}.\n\nProblem: {problem}"
)


def normalize_answer(text):
    text = text.lower().strip()
    text = text.replace("$", "").replace(",", "").replace(" ", "")
    return text


def extract_boxed_answer(text):
    """Grab the content of the LAST \\boxed{...} in the generated text."""
    matches = list(re.finditer(r"\\boxed\{", text))
    if not matches:
        return None
    start = matches[-1].end()
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


def get_covered_ids(*paths):
    covered = set()
    for path in paths:
        if not path or not os.path.exists(path):
            continue
        with open(path) as f:
            for line in f:
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue
                pid = row.get("id")
                if pid is not None:
                    covered.add(pid)
    return covered


def id_to_index(row_id):
    m = re.search(r"(\d+)$", row_id)
    return int(m.group(1)) if m else None


@torch.no_grad()
def generate_sample(model, tokenizer, problem, device, temperature=0.8, max_new_tokens=1024):
    messages = [{"role": "user", "content": PROMPT_TEMPLATE.format(problem=problem)}]
    encoding = tokenizer.apply_chat_template(
        messages, add_generation_prompt=True, return_tensors="pt", return_dict=True
    ).to(device)
    input_ids = encoding["input_ids"]
    attention_mask = encoding.get("attention_mask")

    output = model.generate(
        input_ids,
        attention_mask=attention_mask,
        max_new_tokens=max_new_tokens,
        do_sample=True,
        temperature=temperature,
        top_p=0.95,
        pad_token_id=tokenizer.eos_token_id,
    )
    generated_ids = output[0][input_ids.shape[1]:]
    return tokenizer.decode(generated_ids, skip_special_tokens=True)


def main(existing_file, out_path, num_new_problems, k, start_index):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading teacher model {TEACHER_MODEL} in 4-bit on {device}...")

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(TEACHER_MODEL)
    model = AutoModelForCausalLM.from_pretrained(
        TEACHER_MODEL, quantization_config=bnb_config, device_map="auto"
    )
    model.eval()

    print("Loading MATH-500 dataset...")
    dataset = load_dataset("HuggingFaceH4/MATH-500", split="test")

    covered_ids = get_covered_ids(existing_file, out_path)
    print(f"Already covered (existing + out file so far): {len(covered_ids)} problems")

    # pick the next num_new_problems indices, starting at start_index, not already covered
    candidate_indices = []
    idx = start_index
    while len(candidate_indices) < num_new_problems and idx < len(dataset):
        candidate_id = f"math500-{idx}"
        if candidate_id not in covered_ids:
            candidate_indices.append(idx)
        idx += 1

    print(f"Will generate k={k} samples for {len(candidate_indices)} new problems "
          f"(indices {candidate_indices[0] if candidate_indices else 'N/A'}"
          f"..{candidate_indices[-1] if candidate_indices else 'N/A'})")

    mode = "a" if os.path.exists(out_path) else "w"
    with open(out_path, mode) as f_out:
        for count, idx in enumerate(candidate_indices):
            row = dataset[idx]
            problem = row["problem"]
            gold = row["answer"]
            problem_id = f"math500-{idx}"

            n_correct = 0
            for sample_idx in range(k):
                gen_text = generate_sample(model, tokenizer, problem, device)
                pred = extract_boxed_answer(gen_text)
                correct = (
                    pred is not None
                    and normalize_answer(pred) == normalize_answer(gold)
                )
                n_correct += int(correct)

                out_row = {
                    "id": problem_id,
                    "sample_idx": sample_idx,
                    "answer": gold,
                    "predicted_answer": pred,
                    "reasoning_text": gen_text,
                    "correct": correct,
                }
                f_out.write(json.dumps(out_row) + "\n")
            f_out.flush()

            if (count + 1) % 5 == 0 or count == len(candidate_indices) - 1:
                print(f"  {count + 1}/{len(candidate_indices)} problems done "
                      f"(problem {idx}: {n_correct}/{k} correct)")

    print(f"\nDone. New samples appended to {out_path}")
    print(
        f"Combine with your original file for downstream analysis, e.g.:\n"
        f"  Get-Content {existing_file}, {out_path} | Set-Content traces_teacher_k5_combined.jsonl"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--existing-file", default="traces_teacher_k5.jsonl",
                         help="Your existing trace file, used to skip already-covered problems")
    parser.add_argument("--out", default="traces_teacher_k5_extended.jsonl",
                         help="Output file for the NEW problems' traces (kept separate, resumable)")
    parser.add_argument("--num-new-problems", type=int, default=50)
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--start-index", type=int, default=0,
                         help="MATH-500 index to start scanning from for new (uncovered) problems")
    args = parser.parse_args()
    main(args.existing_file, args.out, args.num_new_problems, args.k, args.start_index)
