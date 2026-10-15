import json

path = "traces_DeepSeek-R1-Distill-Qwen-1-5B_math500.jsonl"
rows = [json.loads(l) for l in open(path)]

fin   = [x for x in rows if x["n_tokens_total"] < 4090]
trunc = [x for x in rows if x["n_tokens_total"] >= 4090]

def acc(g):
    return sum(x["correct"] for x in g) / len(g) if g else 0.0

print("total samples      :", len(rows))
print("finished normally  :", len(fin),   " accuracy: {:.1%}".format(acc(fin)))
print("hit the token cap  :", len(trunc), " accuracy: {:.1%}".format(acc(trunc)))
print("truncated but still produced an answer:",
      sum(1 for x in trunc if x["pred"]))
print("no answer extracted at all:", sum(1 for x in rows if not x["pred"]))
print()
print("reported accuracy (all samples)   : {:.1%}".format(acc(rows)))
print("accuracy among finished traces    : {:.1%}".format(acc(fin)))
