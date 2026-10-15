"""Shared student-model scoring: teacher-forced mean log-prob of the gold answer tokens
given problem + reasoning prefix. Used by every scoring script (identical prompt)."""
import re

import torch
from datasets import load_dataset

STUDENT_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"
PROMPT_TEMPLATE = (
    "Solve the following math problem step by step.\n\nProblem: {problem}\n\n"
    "{prefix_block}Answer: \\boxed{{"
)


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
