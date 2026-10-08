import json
from collections import defaultdict

rows = [json.loads(l) for l in open("traces_DeepSeek-R1-Distill-Qwen-1-5B_math500.jsonl")]
by = defaultdict(lambda: {"n": 0, "trunc": 0, "correct": 0})
for r in rows:
    lv = str(r.get("level"))
    by[lv]["n"] += 1
    by[lv]["trunc"] += 1 if r["n_tokens_total"] >= 4090 else 0
    by[lv]["correct"] += 1 if r["correct"] else 0

print(f'{"level":<10}{"n":>4}{"truncated":>11}{"trunc %":>9}{"acc":>7}')
for lv in sorted(by):
    d = by[lv]
    print(f'{lv:<10}{d["n"]:>4}{d["trunc"]:>11}{d["trunc"]/d["n"]:>9.0%}{d["correct"]/d["n"]:>7.0%}')

ids = sorted({r["id"] for r in rows if r["n_tokens_total"] >= 4090})
print("\ntruncated problem ids:", len(ids))
print(", ".join(ids))
