"""
Matched-problem run of the reasoning teacher (DeepSeek-R1-Distill-Qwen-7B).

Why: the first reasoning run used MATH-500 problems 0-49, while the instruct teacher saw a different 50
problems (only 5 overlap). To compare the teachers fairly, run the reasoning teacher on the SAME problems
the instruct teacher saw.

Same as generation/teacher_trace_generation_reasoning_v2.py (imported, not copied):
  prompt template, answer extraction, think-tag handling, temperature 0.6, top_p 0.95, row fields, id format.
Different on purpose:
  * problems come from --problems-from (any jsonl whose ids end in the MATH-500 index)
  * the k samples of a problem are generated in ONE batched call (num_return_sequences) for speed
  * resumable per sample: rerun the same command after a Colab disconnect and it continues
  * --max-new-tokens defaults to 16384 (v2 used 8192 and lost 18 traces)
  * --precision 4bit (default, matches v2) or bf16 (full precision; needs an A100 or L4 with 24 GB)

Usage:
  python generation/teacher_trace_generation_reasoning_matched.py \
      --problems-from data/cut_instruct_final.jsonl --out data/traces_reasoning_matched_k5.jsonl
"""
import argparse
import json
import os
import re
import sys
import time

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import teacher_trace_generation_reasoning_v2 as v2  # noqa: E402
from datasets import load_dataset  # noqa: E402
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig  # noqa: E402


def indices_from_file(path):
    seen, out = set(), []
    with open(path) as f:
        for line in f:
            if not line.strip():
                continue
            m = re.search(r"(\d+)$", json.loads(line).get("id", ""))
            if m and int(m.group(1)) not in seen:
                seen.add(int(m.group(1)))
                out.append(int(m.group(1)))
    return out


def done_samples(out_path):
    done = {}
    if os.path.exists(out_path):
        with open(out_path) as f:
            for line in f:
                try:
                    r = json.loads(line)
                except json.JSONDecodeError:
                    continue
                m = re.search(r"(\d+)$", r.get("id", ""))
                if m:
                    done.setdefault(int(m.group(1)), set()).add(r.get("sample_idx"))
    return done


def pick_dtype(name):
    """bf16 needs an Ampere-or-newer GPU (L4, A100, RTX 5070). The T4 has no native bf16, so use fp16 there."""
    if name == "bf16":
        return torch.bfloat16
    if name == "fp16":
        return torch.float16
    if torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8:
        return torch.bfloat16
    return torch.float16


def load_model(precision, dtype):
    tok = AutoTokenizer.from_pretrained(v2.MODEL_NAME)
    if precision == "4bit":
        cfg = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                                 bnb_4bit_compute_dtype=dtype, bnb_4bit_use_double_quant=True)
        model = AutoModelForCausalLM.from_pretrained(v2.MODEL_NAME, quantization_config=cfg, device_map="auto")
    else:
        model = AutoModelForCausalLM.from_pretrained(v2.MODEL_NAME, torch_dtype=dtype, device_map="auto")
    model.eval()
    return tok, model


@torch.no_grad()
def generate_n(model, tok, problem, n, max_new_tokens):
    """n independent samples for one problem. Returns [(text, hit_token_limit, n_new_tokens)]."""
    messages = [{"role": "user", "content": v2.USER_TEMPLATE.format(problem=problem)}]
    chat = tok.apply_chat_template(messages, tokenize=True, add_generation_prompt=True,
                                   return_dict=True, return_tensors="pt")
    ids = chat["input_ids"].to(model.device)
    mask = chat["attention_mask"].to(model.device)
    prompt_text = tok.decode(ids[0], skip_special_tokens=False)
    opened_think = "<think>" in prompt_text and "</think>" not in prompt_text

    eos = model.generation_config.eos_token_id
    eos_ids = set(eos) if isinstance(eos, (list, tuple)) else {eos if eos is not None else tok.eos_token_id}
    eos_tensor = torch.tensor(sorted(eos_ids), device=model.device)

    out = model.generate(input_ids=ids, attention_mask=mask, max_new_tokens=max_new_tokens, do_sample=True,
                         temperature=0.6, top_p=0.95, num_return_sequences=n,
                         pad_token_id=tok.eos_token_id)
    new = out[:, ids.shape[1]:]
    results = []
    for row in new:
        hits = torch.isin(row, eos_tensor).nonzero()
        if len(hits):
            n_tok, hit = int(hits[0]) + 1, False
        else:
            n_tok, hit = int(row.shape[0]), True
        text = tok.decode(row[:n_tok], skip_special_tokens=True)
        if opened_think and "<think>" not in text:
            text = "<think>" + text
        results.append((text, hit, n_tok))
    return results


def main(a):
    if not torch.cuda.is_available():
        print("WARNING: no CUDA device found; this will be extremely slow.")
    indices = [int(x) for x in a.indices.split(",")] if a.indices else indices_from_file(a.problems_from)
    if a.limit:
        indices = indices[: a.limit]
    done = done_samples(a.out)
    todo = [(i, [s for s in range(a.k) if s not in done.get(i, set())]) for i in indices]
    todo = [(i, s) for i, s in todo if s]
    print("Problems requested: {}   still to do: {}   precision: {}   max_new_tokens: {}".format(
        len(indices), len(todo), a.precision, a.max_new_tokens))
    if not todo:
        print("Nothing to do.")
        return

    dtype = pick_dtype(a.compute_dtype)
    dtype_name = "bf16" if dtype == torch.bfloat16 else "fp16"
    print("Loading {} ({}, compute {}) ...".format(v2.MODEL_LABEL, a.precision, dtype_name))
    tok, model = load_model(a.precision, dtype)
    dataset = load_dataset("HuggingFaceH4/MATH-500", split="test")

    n_total = n_hit = n_noans = 0
    t0 = time.time()
    with open(a.out, "a") as f_out:
        for n, (idx, samples) in enumerate(todo, start=1):
            row = dataset[idx]
            problem, gold = row["problem"], row["answer"]
            gold_norm = v2.normalize_answer(gold)
            t1 = time.time()
            try:
                results = generate_n(model, tok, problem, len(samples), a.max_new_tokens)
            except torch.cuda.OutOfMemoryError:
                print("  out of memory on a batch of {}; retrying one sample at a time".format(len(samples)))
                torch.cuda.empty_cache()
                results = []
                for _ in samples:
                    results += generate_n(model, tok, problem, 1, a.max_new_tokens)
                    torch.cuda.empty_cache()
            for s, (text, hit, n_tok) in zip(samples, results):
                answer, method = v2.extract_answer(text)
                correct = answer is not None and v2.normalize_answer(answer) == gold_norm
                f_out.write(json.dumps({
                    "id": "math500-reasoning-{}".format(idx), "problem": problem,
                    "reasoning_text": v2.strip_think_tags(text), "answer": gold,
                    "predicted_answer": answer, "extraction_method": method, "correct": correct,
                    "hit_token_limit": hit, "n_generated_tokens": n_tok, "model": v2.MODEL_LABEL,
                    "sample_idx": s, "precision": a.precision, "compute_dtype": dtype_name, "run": "matched"}) + "\n")
                n_total += 1
                n_hit += int(hit)
                n_noans += int(answer is None)
            f_out.flush()
            per = (time.time() - t0) / n
            print("[{}/{}] problem {}  tokens {}  hit_limit {}  {:.0f}s  ETA {:.0f} min".format(
                n, len(todo), idx, [r[2] for r in results], sum(r[1] for r in results),
                time.time() - t1, per * (len(todo) - n) / 60))

    print("\nDone: {} traces written to {}".format(n_total, a.out))
    print("Hit token limit: {}/{}   no extractable answer: {}/{}".format(n_hit, n_total, n_noans, n_total))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--problems-from", help="jsonl whose ids end in the MATH-500 index")
    ap.add_argument("--indices", help="comma-separated MATH-500 indices instead of --problems-from")
    ap.add_argument("--out", default="data/traces_reasoning_matched_k5.jsonl")
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--max-new-tokens", type=int, default=16384)
    ap.add_argument("--precision", choices=["4bit", "bf16"], default="4bit")
    ap.add_argument("--compute-dtype", choices=["auto", "bf16", "fp16"], default="auto",
                    help="auto = bf16 on Ampere or newer, fp16 on a T4")
    ap.add_argument("--limit", type=int, default=0, help="only the first N problems (for a quick test)")
    args = ap.parse_args()
    if not args.problems_from and not args.indices:
        ap.error("give --problems-from or --indices")
    main(args)
