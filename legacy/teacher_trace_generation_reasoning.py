"""
Generate teacher reasoning traces from a REASONING-TUNED model, for direct
comparison against the existing Qwen2.5-7B-Instruct (instruct-tuned) traces.

Why this script exists: every finding so far (answer-position bucketing,
adaptive-cut compression, signal-retention scoring, the difficulty-stratified
versions of all three) was produced with an instruct-tuned teacher. This
generates a matched dataset with a reasoning-tuned teacher of the same scale
(DeepSeek-R1-Distill-Qwen-7B, 7B params, same family lineage as Qwen2.5-7B)
so the existing analysis scripts can be rerun unchanged and compared side by
side -- the question is whether "where does the trace first state the
answer" and "does adaptive cutting preserve signal" hold for a model that
was actually trained to produce long chain-of-thought, not just prompted to.

Key differences from teacher_trace_generation_local.py (the instruct-teacher
version):

1. DeepSeek-R1-Distill models wrap their reasoning in <think>...</think>
   tags. We KEEP these tags in reasoning_text (stripped of the tag markup
   itself, content preserved) -- the think block IS the reasoning trace we
   want to analyze. If the model's chat template auto-opens the <think>
   block as part of the generation prompt, we prepend it back so the stored
   trace is self-consistent.
2. DeepSeek's own model card advises AGAINST greedy decoding for R1-distill
   models (it causes repetition loops) and recommends temperature=0.6,
   top_p=0.95 -- used here instead of the instruct-teacher's settings.
3. Traces from reasoning-tuned models run much longer, so max_new_tokens is
   raised substantially and progress is logged per-sample (not just
   per-problem) since a single k=5 problem can take minutes.

Schema written matches the existing pipeline exactly (same field names
consumed by answer_position_analysis.py / adaptive_cut_analysis.py /
difficulty_stratified_analysis.py / score_by_difficulty.py):
    id, problem, reasoning_text, answer, correct, model, sample_idx

Usage:
    python teacher_trace_generation_reasoning.py --out traces_reasoning_teacher_k5.jsonl --num-problems 50 --k 5

Resumable: rerunning with the same --out skips problem indices already
present in that file.
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

# DeepSeek's model card: do not use a system prompt, and put the full
# instruction in the user turn. Asking explicitly for \boxed{} keeps the
# answer extraction consistent with the rest of the pipeline.
USER_TEMPLATE = (
    "Solve the following math problem step by step. "
    "Put your final answer in \\boxed{{}}.\n\nProblem: {problem}"
)


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
    """Finds the LAST \\boxed{...} in text, brace-depth aware (so nested
    braces inside the answer, e.g. \\boxed{\\frac{1}{2}}, don't truncate
    early). Returns None if no boxed answer is found."""
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


def strip_think_tags(text):
    """Removes the literal <think> / </think> markers but keeps their
    content, so reasoning_text reads as continuous prose for the sentence-
    splitting logic in the analysis scripts."""
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

    # If the chat template already opened a <think> block as part of the
    # generation prompt (DeepSeek-R1-Distill templates do this), capture
    # that prefix so it's preserved in the stored trace.
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
    completion = tokenizer.decode(new_tokens, skip_special_tokens=True)

    if template_opened_think and "<think>" not in completion:
        completion = "<think>" + completion

    return completion


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

    print(f"Generating k={k} samples each for {len(target_indices)} problems -> {out_path}")
    mode = "a" if os.path.exists(out_path) else "w"
    with open(out_path, mode) as f_out:
        for n, idx in enumerate(target_indices, start=1):
            row = dataset[idx]
            problem_text = row["problem"]
            gold_raw = row["answer"]
            gold_norm = normalize_answer(gold_raw)

            for sample_idx in range(k):
                completion = generate_sample(model, tokenizer, problem_text, device, max_new_tokens)
                boxed = extract_boxed_answer(completion)
                correct = (
                    boxed is not None and normalize_answer(boxed) == gold_norm
                )
                reasoning_text = strip_think_tags(completion)

                out_row = {
                    "id": f"math500-reasoning-{idx}",
                    "problem": problem_text,
                    "reasoning_text": reasoning_text,
                    "answer": gold_raw,
                    "predicted_answer": boxed,
                    "correct": correct,
                    "model": MODEL_LABEL,
                    "sample_idx": sample_idx,
                }
                f_out.write(json.dumps(out_row) + "\n")
                f_out.flush()
                print(
                    f"  [{n}/{len(target_indices)}] problem idx={idx} sample={sample_idx} "
                    f"correct={correct} len={len(reasoning_text.split())}w"
                )

    print(f"\nDone. Traces written to {out_path}")
    print(
        "\nNext: rerun the existing analysis scripts against this file, e.g.\n"
        f"  python answer_position_analysis.py --trace-file {out_path}\n"
        f"  python adaptive_cut_analysis.py --trace-file {out_path} --out adaptive_cut_traces_reasoning.jsonl\n"
        "and compare the bucketing/compression numbers against the instruct-teacher results."
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="traces_reasoning_teacher_k5.jsonl")
    parser.add_argument("--num-problems", type=int, default=50)
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument(
        "--max-new-tokens", type=int, default=3072,
        help="reasoning-tuned models produce much longer traces than instruct models; raise if traces look truncated"
    )
    args = parser.parse_args()
    main(args.out, args.num_problems, args.k, args.start_index, args.max_new_tokens)
