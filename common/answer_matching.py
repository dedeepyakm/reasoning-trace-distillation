"""Shared answer-matching logic. EVERY analysis script imports from here so a
fix lands everywhere (the Oct-14 audit found three scripts carrying three
different copies of this logic).

- strip_wrappers / answers_match : decide whether a model's answer equals gold
  (handles \text{}, degrees, %, \dfrac, thousands commas, trailing units;
  sentence-style fallback answers are judged by whether they contain gold).
- sentence_mentions_answer       : does a sentence really STATE an answer?
  Numeric answers need clean digit boundaries and a result cue before them
  ('=', 'is', 'answer', 'has', ...), so '9' never matches inside '196'.
- is_restatement                 : sentence mostly re-quotes the problem.
- cut_target                     : the gold-free target for the adaptive cut --
  the MODEL'S OWN final answer (not the gold).
"""
import re

CUE_BEFORE = re.compile(
    r"(=|\bis\b|\bare\b|\bequals?\b|\bbe\b|\banswer\b|\bgives?\b|\bgets?\b|\bgot\b"
    r"|\bhas\b|\bhave\b|\\approx|\\boxed\{|\\u2192)[\s:\(\[\\\{]*$"
)


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


def is_restatement(sent, problem):
    """Sentence mostly re-quotes the problem statement (words already in it)."""
    words = re.findall(r"[a-z0-9]+", sent.lower())
    if len(words) < 6:
        return False
    pw = set(re.findall(r"[a-z0-9]+", problem.lower()))
    return sum(w in pw for w in words) / len(words) >= 0.8


def cut_target(r, correct, gold):
    """Gold-free target: what the MODEL itself concluded."""
    pred, method = r.get("predicted_answer"), r.get("extraction_method")
    if pred is None:
        return None
    if method == "boxed":
        return pred
    if correct:
        return gold
    nums = re.findall(r"-?\d+(?:\.\d+)?", pred.replace(",", ""))
    return nums[-1] if nums else None
