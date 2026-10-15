import argparse
import json
import re
from collections import defaultdict

CUE_BEFORE = re.compile(
    r"(=|\bis\b|\bare\b|\bequals?\b|\bbe\b|\banswer\b|\bgives?\b|\bgets?\b|\bgot\b"
    r"|\bhas\b|\bhave\b|\\approx|\\boxed\{|\\u2192)[\s:\(\[\\\{]*$"
)
TIER_NAMES = {1: "Easy", 2: "Easy", 3: "Medium", 4: "Hard", 5: "Hard"}
TIER_ORDER = ["Easy", "Medium", "Hard"]


def normalize_for_search(text):
    text = text.lower().strip()
    text = text.replace("$", "").replace(",", "")
    return re.sub(r"\s+", " ", text)


def split_sentences(text):
    parts = re.split(r"(?<=[.?!])\s+", text.strip())
    return [p for p in parts if p]


def is_purely_numeric(s):
    return re.match(r"^-?\d+(\.\d+)?$", s.replace(" ", "")) is not None


def sentence_mentions_answer(norm_sent, norm_gold, need_cue=None):
    """True only for a real statement of the gold answer: numeric answers need
    clean digit boundaries AND a result cue right before them ('=', 'is',
    'answer', 'has', ...), so '9' does not match inside '196', and numbers
    that are only listed or copied from the problem do not count."""
    if not norm_gold:
        return False
    numeric = is_purely_numeric(norm_gold)
    if need_cue is None:
        need_cue = numeric or len(norm_gold) <= 3
    if numeric:
        pattern = r"(?<![\d.])" + re.escape(norm_gold.replace(" ", "")) + r"(?!\d|\.\d)"
    else:
        pattern = re.escape(norm_gold)
    for m in re.finditer(pattern, norm_sent):
        if not need_cue:
            return True
        if CUE_BEFORE.search(norm_sent[: m.start()]):
            return True
    return False


def first_mention_index(sentences, gold):
    norm_gold = normalize_for_search(strip_wrappers(gold))
    for i, sent in enumerate(sentences):
        if sentence_mentions_answer(normalize_for_search(sent), norm_gold):
            return i
    return None


def strip_wrappers(s):
    s = s.strip()
    s = re.sub(r"\\text(?:bf)?\{([^{}]*)\}", r"\1", s)
    for junk in ["\\!", "\\,", "\\;", "\\ ", "\\left", "\\right"]:
        s = s.replace(junk, "")
    s = s.replace("dfrac", "frac").replace("tfrac", "frac")
    for junk in ["^\\circ", "^{\\circ}", "\\circ", "\\degree", "\\%", "%", "\\$", "$"]:
        s = s.replace(junk, "")
    s = re.sub(r"(?<=\d),(?=\d{3})", "", s)
    return s.replace(" ", "").rstrip(".")


def num_core(s):
    m = re.fullmatch(r"(-?\d+(?:\.\d+)?)[A-Za-z]*", s)
    return float(m.group(1)) if m else None


def answers_match(pred, gold, method):
    if pred is None:
        return False
    g = strip_wrappers(gold)
    if method == "fallback_text":
        # pred is a phrase like 'The perimeter ... is 42 inches'
        return sentence_mentions_answer(normalize_for_search(pred), normalize_for_search(g), need_cue=False)
    p = strip_wrappers(pred)
    if p == g:
        return True
    pc, gc = num_core(p), num_core(g)
    return pc is not None and gc is not None and abs(pc - gc) < 1e-9


def load_level_lookup():
    from datasets import load_dataset
    ds = load_dataset("HuggingFaceH4/MATH-500", split="test")
    out = {}
    for i, row in enumerate(ds):
        try:
            out[i] = int(str(row.get("level")).strip().split()[-1])
        except (ValueError, IndexError):
            out[i] = None
    return out


def pct(a, b):
    return 100.0 * a / b if b else 0.0


def main(trace_file, out_path, with_difficulty):
    levels = load_level_lookup() if with_difficulty else None
    rows_out = []
    flips_up = flips_down = n_relabelled = 0
    acc = defaultdict(lambda: [0, 0])
    buckets = defaultdict(lambda: defaultdict(lambda: [0, 0]))
    fracs = {"correct": [], "incorrect": []}
    never = [0, 0]

    with open(trace_file) as f:
        for line in f:
            r = json.loads(line)
            text, gold, rid = r.get("reasoning_text"), r.get("answer"), r.get("id")
            if text is None or gold is None or rid is None:
                continue
            correct = bool(r.get("correct"))
            if "predicted_answer" in r:
                new = answers_match(r["predicted_answer"], gold, r.get("extraction_method"))
                n_relabelled += 1
                if new and not correct:
                    flips_up += 1
                if correct and not new:
                    flips_down += 1
                correct = new

            tier = "All"
            if levels is not None:
                m = re.search(r"(\d+)$", rid)
                tier = TIER_NAMES.get(levels.get(int(m.group(1)))) if m else None
                if tier is None:
                    continue

            sents = split_sentences(text)
            idx = first_mention_index(sents, gold) if sents else None
            if idx is None:
                bucket, kept, frac = "never_explicit", text, 1.0
                never[0 if correct else 1] += 1
            else:
                bucket = "last_two" if idx >= len(sents) - 2 else "early_or_middle"
                kept = " ".join(sents[: idx + 1])
                frac = len(kept.split()) / max(1, len(text.split()))
                fracs["correct" if correct else "incorrect"].append(frac)

            acc[tier][0] += int(correct)
            acc[tier][1] += 1
            buckets[tier][bucket][0 if correct else 1] += 1
            rows_out.append({
                "id": rid, "correct": correct, "answer": gold, "difficulty_tier": tier,
                "never_explicit": idx is None, "cut_sentence_idx": idx,
                "total_sentences": len(sents), "frac_words_kept": round(frac, 4),
                "adaptive_truncated_text": kept,
            })

    with open(out_path, "w") as f:
        for r in rows_out:
            f.write(json.dumps(r) + "\n")

    total = sum(v[1] for v in acc.values())
    print(f"Analyzed {total} traces from {trace_file}")
    if n_relabelled:
        print(f"Correctness recomputed for {n_relabelled} traces: "
              f"{flips_up} incorrect->correct, {flips_down} correct->incorrect")
    print("\nACCURACY (trace level)")
    for t in (TIER_ORDER if with_difficulty else ["All"]):
        c, n = acc[t]
        print(f"  {t:<8} {c}/{n} = {pct(c, n):.1f}%")

    print("\nANSWER POSITION (first real statement of the gold answer)")
    print(f"  {'Tier':<8}{'Bucket':<18}{'Correct':>8}{'Incorrect':>10}{'Total':>7}{'%Correct':>10}")
    allb = defaultdict(lambda: [0, 0])
    for t in (TIER_ORDER if with_difficulty else ["All"]):
        for b in ["early_or_middle", "last_two", "never_explicit"]:
            c, i = buckets[t][b]
            allb[b][0] += c
            allb[b][1] += i
            print(f"  {t:<8}{b:<18}{c:>8}{i:>10}{c + i:>7}{pct(c, c + i):>9.1f}%")
    if with_difficulty:
        for b in ["early_or_middle", "last_two", "never_explicit"]:
            c, i = allb[b]
            print(f"  {'ALL':<8}{b:<18}{c:>8}{i:>10}{c + i:>7}{pct(c, c + i):>9.1f}%")

    print("\nADAPTIVE CUT: average % of words kept (traces with a real cut point only)")
    for label in ["correct", "incorrect"]:
        v = fracs[label]
        print(f"  {label:<10} N={len(v):<4} {100 * sum(v) / len(v) if v else 0:.1f}% kept")
    print(f"\nNever-explicit traces (kept in full, excluded above): correct={never[0]} incorrect={never[1]}")
    print(f"Cut traces saved to {out_path}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--trace-file", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--with-difficulty", action="store_true")
    a = ap.parse_args()
    main(a.trace_file, a.out, a.with_difficulty)
