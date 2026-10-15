r"""Shared answer-matching logic. EVERY analysis script imports from here so a
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


def strip_wrappers(s, keep_spaces=False):
    s = s.strip()
    s = re.sub(r"\\text(?:bf)?\{([^{}]*)\}", r"\1", s)
    for junk in ["\\!", "\\,", "\\;", "\\ ", "\\left", "\\right"]:
        s = s.replace(junk, "")
    s = s.replace("dfrac", "frac").replace("tfrac", "frac")
    for junk in ["^\\circ", "^{\\circ}", "\\circ", "\\degree", "\\%", "%", "\\$", "$"]:
        s = s.replace(junk, "")
    s = re.sub(r"(?<=\d),(?=\d{3})", "", s)
    s = (s if keep_spaces else s.replace(" ", "")).strip().rstrip(".")
    m = re.match(r"^[A-Za-z]=([^=]+)$", s)  # 'x=5' -> '5'
    s = m.group(1) if m else s
    return re.sub(r"^(-?)\.(\d)", r"\g<1>0.\2", s)  # '.35' -> '0.35'


def num_core(s):
    m = re.fullmatch(r"(-?\d+(?:\.\d+)?)[A-Za-z]*", s)
    return float(m.group(1)) if m else None


def answers_match(pred, gold, method):
    if pred is None:
        return False
    g = strip_wrappers(gold)
    if method == "fallback_text":
        # pred is a phrase like 'The perimeter ... is 42 inches' or just '6 - 5i'
        npred = normalize_for_search(pred)
        g_sp = normalize_for_search(strip_wrappers(gold, keep_spaces=True))
        if sentence_mentions_answer(npred, g_sp, need_cue=False):
            return True
        if not is_purely_numeric(g):          # spacing differences, e.g. '6 - 5i' vs '6-5i'
            return sentence_mentions_answer(npred.replace(" ", ""), normalize_for_search(g).replace(" ", ""), need_cue=False)
        return False
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


DOUBT = re.compile(r"\b(wait|hmm|not sure|let me (?:check|verify|double|re)|seems? (?:a bit )?(?:off|wrong)|"
                   r"that can't be|doesn't seem)\b", re.IGNORECASE)
STRONG_CUE = re.compile(r"\b(answer|final|therefore|thus|hence|result|conclude|conclusion)\b", re.IGNORECASE)


def number_in_problem(target_norm, problem):
    """Numeric target that also appears as a standalone number in the problem text."""
    if not is_purely_numeric(target_norm) or not problem:
        return False
    return re.search(r"(?<![\d.])" + re.escape(target_norm) + r"(?!\d|\.\d)", normalize_for_search(problem)) is not None


def is_commit_sentence(sent, target_norm, problem):
    """Stricter than sentence_mentions_answer: does this sentence COMMIT to the answer?
    Rejects (a) questions ('does that mean n = 6 is the answer?'), (b) sentences that flag
    doubt ('Wait, that seems a bit off'), and (c) numbers that also occur in the problem
    statement unless the sentence has a strong cue (answer / therefore / final / ...)."""
    st = sent.strip()
    if st.endswith("?"):
        return False
    if DOUBT.search(st):
        return False
    if not sentence_mentions_answer(normalize_for_search(st), target_norm):
        return False
    if number_in_problem(target_norm, problem) and not STRONG_CUE.search(st):
        return False
    return True


def all_boxed(text):
    """Every \\boxed{...} in the text (brace-depth aware), in order."""
    out, i = [], 0
    while True:
        i = text.find("\\boxed{", i)
        if i < 0:
            return out
        j, depth = i + 7, 1
        while j < len(text) and depth:
            depth += {"{": 1, "}": -1}.get(text[j], 0)
            j += 1
        if depth == 0:
            out.append(text[i + 7:j - 1])
        i = j


def multi_boxed_match(text, gold):
    """Final line gives several boxed items ('\\boxed{3}, \\boxed{5} and \\boxed{7}') and gold is the
    list '3, 5, 7': compare as sets. Only the last 400 characters are used."""
    items = all_boxed(text[-400:])
    if len(items) < 2:
        return False
    g = [strip_wrappers(x) for x in re.split(r",|\band\b", gold) if strip_wrappers(x)]
    m = [strip_wrappers(x) for x in items]
    return len(g) >= 2 and set(g) == set(m)
