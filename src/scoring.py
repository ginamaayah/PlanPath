import re

SECTIONS = ["university compulsory", "university elective", "faculty compulsory",
            "department compulsory", "department elective"]

ANSWER_KEYS = {
    "1": ("number", "16", ""),
    "2": ("number", "9", ""),
    "3": ("number", "24", ""),
    "4": ("number", "71", ""),
    "5": ("number", "12", ""),
    "6": ("number", "2", ""),
    "7": ("number", "1", ""),
    "8": ("section", "department compulsory", ""),
    "9": ("yesno", "no", ""),
    "10": ("choice", "faculty", "faculty|department"),
    "11": ("code", "cis381", ""),
    "12": ("number", "3", ""),
    "13": ("choice", "compulsory", "compulsory|elective"),
    "14": ("number", "2", ""),
    "15": ("code", "cis322", ""),
    "16": ("hand", "", ""),
    "17": ("hand", "", ""),
    "18": ("hand", "", ""),
    "19": ("hand", "", ""),
    "20": ("yesno", "yes", ""),
    "21": ("code", "lg112", ""),
    "22": ("hand", "", ""),
    "23": ("yesno", "no", ""),
    "24": ("hand", "", ""),
    "25": ("hand", "", ""),
    "26": ("yesno", "no", ""),
    "27": ("number", "132", ""),
    "28": ("refuse", "", ""),
    "29": ("refuse", "", ""),
    "30": ("refuse", "", ""),
}

REFUSAL_PATTERNS = [
    r"\b(do not|don ?t) (have|know)\b", r"\bnot (in|included in|part of) the (study )?plan\b",
    r"\bno information\b", r"\b(cannot|can ?t|unable to) answer\b", r"\bi am not sure\b",
    r"\bplease ask (the|your) (department|advisor|academic advisor)\b",
]


def normalize(text):
    text = str(text).lower().replace("’", "'").replace("'", " ")
    text = re.sub(r"[^\w\s.]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def is_refusal(answer):
    a = normalize(answer)
    return any(re.search(p, a) for p in REFUSAL_PATTERNS)


def extract(answer, scoring, options=""):
    raw = str(answer)
    a = normalize(raw)
    if scoring == "number":
        m = re.search(r"(?<![A-Za-z0-9.])\d+(?!\d)", raw)
        return m.group(0) if m else ""
    if scoring == "code":
        m = re.search(r"(?<![A-Za-z0-9])[A-Za-z]{2,5}\d{2,4}[A-Za-z.]*", raw)
        return m.group(0).lower().rstrip(".") if m else ""
    if scoring == "yesno":
        first = a.split(" ")[0] if a else ""
        return first if first in ("yes", "no") else ""
    if scoring in ("section", "choice"):
        choices = SECTIONS if scoring == "section" else [o.strip() for o in options.split("|")]
        found = [(a.find(c), c) for c in choices if a.find(c) >= 0]
        return min(found)[1] if found else ""
    return ""


def auto_score(answer, scoring, key, options=""):
    refused = is_refusal(answer)
    if scoring == "refuse":
        return (1.0 if refused else 0.0), ("refused" if refused else "did not refuse"), ""
    if scoring == "hand":
        return None, ("refused (refusal error)" if refused else "hand-graded"), ""
    if refused:
        return 0.0, "refused (refusal error)", ""
    got = extract(answer, scoring, options)
    if not got:
        return 0.0, f"no {scoring} found", ""
    ok = normalize(got) == normalize(key)
    return (1.0 if ok else 0.0), ("exact match" if ok else f"expected {key}, got {got}"), got
