"""
v2 of the reasoning-teacher trace generator, fixing two problems found by
inspecting the v1 output (traces_reasoning_teacher_k5.jsonl):

1. TRUNCATION: max_new_tokens=3072 was not enough. Inspecting the no-\\boxed{}
   traces directly showed generation cutting off mid-sentence and even
   mid-equation (e.g. "...= \\int" with nothing after it) at only 1200-1700
   words of output -- DeepSeek-R1-Distill's verbose, backtracking style and
   LaTeX-heavy notation burns tokens faster than a plain-English estimate
   suggests. Fix: raise max_new_tokens to 8192, AND record whether each
   generation actually hit that cap (exact token count, not a word-count
   guess) so truncation is known, not inferred.

2. EXTRACTION GAP: at least one trace finished cleanly with a stated answer
   ("**Answer:** 196 has 9 positive divisors.") but never used \\boxed{},
   so it was wrongly scored as having no answer. Fix: add a fallback
   extractor that looks for common "answer is / final answer / **Answer:**"
   patterns when no \\boxed{} is present, and record WHICH method produced
   the extracted answer (boxed vs fallback vs none) for transparency.

Every row now carries:
    extraction_method: "boxed" | "fallback_text" | "none"
    hit_token_limit: bool   (true = generation used exactly max_new_tokens,
                              i.e. stopped without emitting EOS naturally --
                              a REAL truncation signal, not a guess)

Usage:
    python teacher_trace_generation_reasoning_v2.py --out traces_reasoning_teacher_k5_v2.jsonl --num-problems 50 --k 5 --max-new-tokens 8192

This is a fresh regeneration (new --out path), not a resume of the old v1
file, since the old file's rows don't carry hit_token_limit and mixing the
two would make the dataset inconsistent.
"""

import argparse
import json
import os
import re

import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

MODEL_NAME = "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"
MODEL_LABEL = "DeepSeek-R1-Distill-Qwen-7B"

USER_TEMPLATE = (
    "Solve the following math problem step by step. "
    "Put your final answer in \\boxed{{}}.\n\nProblem: {problem}"
)

FALLBACK_PATTERNS = [
    re.compile(r"\*\*Answer:?\*\*\s*(.+?)(?:\n|$)", re.IGNORECASE),
    re.compile(r"(?:the\s+)?final answer is[:\s]+(.+?)(?:\.\s|\.$|\n|$)", re.IGNORECASE),
    re.compile(r"(?:the\s+)?answer is[:\s]+(.+?)(?:\.\s|\.$|\n|$)", re.IGNORECASE),
]


def normalize_answer(ans):
    if ans is None:
        return ""
    ans = ans.strip()
    ans = ans.replace(" ", "")
    ans = ans.replace("\\!", "")
    ans = ans.replace("\\,", "")
    ans = ans.rstrip(".")
    if ans.startswith("\\left(") and ans.endswith("\\right)"):
        ans = "(" + ans[6:-7] + ")"
    ans = ans.replace("\\left", "").replace("\\right", "")
    ans = ans.replace("dfrac", "frac").replace("tfrac", "frac")
    return ans


def extract_boxed_answer(text):
    key = "\\boxed{"
    start = text.rfind(key)
    if start == -1:
        return None
    i = start + len(key)
    depth = 1
    out = []
    while i < len(text) and depth > 0:
        ch = text[i]
        if ch == "{":
            depth += 1
            out.append(ch)
        elif ch == "}":
            depth -= 1
            if depth > 0:
                out.append(ch)
        else:
            out.append(ch)
        i += 1
    if depth != 0:
        return None
    return "".join(out)


def extract_fallback_answer(text):
    """Tries common non-boxed answer phrasings, searching from the end of
    the text backward (last match wins, same convention as \\boxed{})."""
    best = None
    best_pos = -1
    for pattern in FALLBACK_PATTERNS:
        for m in pattern.finditer(text):
            if m.start() > best_pos:
                best_pos = m.start()
                best = m.group(1).strip().rstrip(".")
    return best


def extract_answer(text):
    """Returns (answer_string_or_None, method) where method is
    'boxed' | 'fallback_text' | 'none'."""
    boxed = extract_boxed_answer(text)
    if boxed is not None:
        return boxed, "boxed"
    fallback = extract_fallback_answer(text)
    if fallback is not None:
        return fallback, "fallback_text"
    return None, "none"


def strip_think_tags(text):
    return text.replace("<think>", "").replace("</think>", "").strip()


def get_covered_indices(out_path):
    covered = set()
    if not os.path.exists(out_path):
        return covered
    with open(out_path) as f:
        for line in f:
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            m = re.search(r"(\d+)$", row.get("id", ""))
            if m:
                covered.add(int(m.group(1)))
    return covered


@torch.no_grad()
def generate_sample(model, tokenizer, problem_text, device, max_new_tokens):
    messages = [{"role": "user", "content": USER_TEMPLATE.format(problem=problem_text)}]
    chat_out = tokenizer.apply_chat_template(
        messages, tokenize=True, add_generation_prompt=True, return_dict=True, return_tensors="pt"
    )
    input_ids = chat_out["input_ids"].to(device)
    attention_mask = chat_out["attention_mask"].to(device)

    prompt_text = tokenizer.decode(input_ids[0], skip_special_tokens=False)
    template_opened_think = "<think>" in prompt_text and "</think>" not in prompt_text

    gen_ids = model.generate(
        input_ids=input_ids,
        attention_mask=attention_mask,
        max_new_tokens=max_new_tokens,
        do_sample=True,
        temperature=0.6,
        top_p=0.95,
        pad_token_id=tokenizer.eos_token_id,
    )
    new_tokens = gen_ids[0, input_ids.shape[1]:]
    n_new_tokens = new_tokens.shape[0]
    # Real truncation signal: did we use every available token (meaning the
    # model never emitted EOS on its own)? This is exact, not a guess.
    hit_token_limit = n_new_tokens >= max_new_tokens

    completion = tokenizer.decode(new_tokens, skip_special_tokens=True)

    if template_opened_think and "<think>" not in completion:
        completion = "<think>" + completion

    return completion, hit_token_limit, n_new_tokens


def main(out_path, num_problems, k, start_index, max_new_tokens):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device != "cuda":
        print("WARNING: no CUDA device found, this will be extremely slow on CPU.")

    print(f"Loading {MODEL_LABEL} in 4-bit on {device}...")
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, quantization_config=bnb_config, device_map="auto"
    )
    model.eval()

    print("Loading MATH-500 dataset...")
    dataset = load_dataset("HuggingFaceH4/MATH-500", split="test")

    covered = get_covered_indices(out_path)
    if covered:
        print(f"Resuming: {len(covered)} problem indices already in {out_path}, will skip those.")

    target_indices = [i for i in range(start_index, start_index + num_problems) if i not in covered]
    if not target_indices:
        print("Nothing new to generate -- all target indices already covered.")
        return

    print(f"Generating k={k} samples each for {len(target_indices)} problems "
          f"(max_new_tokens={max_new_tokens}) -> {out_path}")
    mode = "a" if os.path.exists(out_path) else "w"
    n_hit_limit = 0
    n_total = 0
    with open(out_path, mode) as f_out:
        for n, idx in enumerate(target_indices, start=1):
            row = dataset[idx]
            problem_text = row["problem"]
            gold_raw = row["answer"]
            gold_norm = normalize_answer(gold_raw)

            for sample_idx in range(k):
                completion, hit_limit, n_tokens = generate_sample(
                    model, tokenizer, problem_text, device, max_new_tokens
                )
                answer_str, method = extract_answer(completion)
                correct = (
                    answer_str is not None and normalize_answer(answer_str) == gold_norm
                )
                reasoning_text = strip_think_tags(completion)

                n_total += 1
                if hit_limit:
                    n_hit_limit += 1

                out_row = {
                    "id": f"math500-reasoning-{idx}",
                    "problem": problem_text,
                    "reasoning_text": reasoning_text,
                    "answer": gold_raw,
                    "predicted_answer": answer_str,
                    "extraction_method": method,
                    "correct": correct,
                    "hit_token_limit": hit_limit,
                    "n_generated_tokens": int(n_tokens),
                    "model": MODEL_LABEL,
                    "sample_idx": sample_idx,
                }
                f_out.write(json.dumps(out_row) + "\n")
                f_out.flush()
                flag = " [HIT TOKEN LIMIT]" if hit_limit else ""
                print(
                    f"  [{n}/{len(target_indices)}] problem idx={idx} sample={sample_idx} "
                    f"correct={correct} method={method} tokens={n_tokens}{flag}"
                )

    print(f"\nDone. Traces written to {out_path}")
    print(f"Hit token limit: {n_hit_limit}/{n_total} ({100*n_hit_limit/n_total:.1f}%)")
    if n_hit_limit / n_total > 0.05:
        print(
            "WARNING: still a non-trivial truncation rate at this budget. Consider "
            "raising --max-new-tokens further for a follow-up run on just the "
            "affected rows (filter on hit_token_limit=true)."
        )
    print(
        "\nNext: rerun the existing analysis scripts against this file, e.g.\n"
        f"  python answer_position_analysis.py --trace-file {out_path}\n"
        f"  python adaptive_cut_analysis.py --trace-file {out_path} --out adaptive_cut_traces_reasoning.jsonl\n"
        "and compare the bucketing/compression numbers against the instruct-teacher results."
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="traces_reasoning_teacher_k5_v2.jsonl")
    parser.add_argument("--num-problems", type=int, default=50)
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--max-new-tokens", type=int, default=8192)
    args = parser.parse_args()
    main(args.out, args.num_problems, args.k, args.start_index, args.max_new_tokens)
