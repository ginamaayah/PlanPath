import argparse
import csv
import difflib
import json
import os
import random
import re
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.common import load_config
from src.course_data import group_text, join_list, label, load_courses

REFUSAL = "I don't have that information. The study plan does not include it; please ask the department."

TEMPLATES = {
    "hours": ["How many credit hours is {name}?", "How many credit hours is {code}?",
              "how many hours is {lname}?", "What is the credit hour count of {code} {name}?"],
    "code": ["What is the course code for {name}?", "what's the code of {lname}?",
             "Which code does {name} have in the CIS plan?"],
    "name": ["What is {code}?", "Which course is {code}?", "what course has the code {code}?"],
    "section": ["Which requirement group does {code} {name} belong to?", "Where is {name} in the study plan?",
                "which part of the plan is {code} in?"],
    "type": ["Is {name} compulsory or elective?", "is {code} a required course?",
             "Do I have to take {code} {name}?"],
    "teaching": ["How is {code} taught?", "Is {name} taught on campus or online?",
                 "what is the teaching method of {code}?"],
    "prereq": ["What do I need before taking {code}?", "What are the prerequisites for {code} {name}?",
               "does {lname} have a prerequisite?", "What must I study before {name}?"],
    "equivalent": ["Which course is equivalent to {code}?", "What is the equivalent course of {code} {name}?",
                   "is there an equivalent for {code}?"],
}

REFUSAL_TEMPLATES = [
    "When is the final exam of {code}?",
    "Which room is {name} taught in?",
    "What time is the {code} lecture?",
    "How much does one credit hour of {code} cost?",
    "How many students can register in {code}?",
    "What grade do I need to pass {name}?",
    "What is the attendance policy for {code}?",
    "When is the last day to drop {code}?",
    "What is the email of the {code} coordinator?",
    "Which textbook edition is the cheapest for {name}?",
]


def hours_text(h):
    return f"{h} credit hour" + ("" if h == 1 else "s")


def prereq_answer(rec, names):
    code = rec["code"]
    parts = []
    if rec["prereq_study"]:
        parts.append(f"you must have studied {join_list([group_text(g, names) for g in rec['prereq_study']])} first")
    if rec["prereq_pass"]:
        parts.append(f"you must have passed {join_list([group_text(g, names) for g in rec['prereq_pass']])} first")
    if rec["prereq_or_concurrent"]:
        parts.append(f"you must take {join_list([group_text(g, names) for g in rec['prereq_or_concurrent']])} "
                     f"before it or in the same semester")
    if rec["concurrent"]:
        parts.append(f"it must be taken in the same semester as "
                     f"{join_list([group_text(g, names) for g in rec['concurrent']])}")
    if not parts:
        return f"No prerequisite: {code} has no prerequisite in the study plan."
    return f"For {code}, " + "; ".join(parts) + "."


def course_pairs(rec, names, rng):
    code, name = rec["code"], rec["name"]
    fields = {"code": code, "name": name, "lname": name.lower()}
    kind = "compulsory" if rec["compulsory"] else "elective"
    answers = {
        "hours": f"{hours_text(rec['hours'])} ({code} {name}).",
        "code": f"{code} is the course code for {name}.",
        "name": f"{code} is {name} ({hours_text(rec['hours'])}).",
        "section": f"{rec['section']} ({code} {name}).",
        "type": (f"{'Yes' if rec['compulsory'] else 'No'}; {code} is {kind}: it is in the {rec['section']}."),
        "teaching": f"{code} {name} is taught {rec['teaching']}.",
        "prereq": prereq_answer(rec, names),
    }
    eq = [c for c in rec["equivalents"] if c != code]
    if eq:
        answers["equivalent"] = f"{join_list([label(c, names) for c in eq])} is listed as equivalent to {code}."
    pairs = []
    for ptype, answer in answers.items():
        template = rng.choice(TEMPLATES[ptype])
        question = template.format(**fields)
        if ptype == "type" and template.startswith("Is {name} compulsory or elective"):
            answer = f"{kind.capitalize()}; {code} is in the {rec['section']}."
        pairs.append({"type": ptype, "course": code, "question": question, "answer": answer})
    return pairs


def eligibility_pairs(rec, names, rng):
    groups = rec["prereq_study"] + rec["prereq_pass"]
    if not groups or rec["concurrent"] or rec["prereq_or_concurrent"]:
        return []
    code = rec["code"]
    taken = [g[0] for g in groups]
    out = [{
        "type": "eligible_yes", "course": code,
        "question": f"I have passed {join_list(taken)}. Can I register for {code}?",
        "answer": f"Yes; you have completed the prerequisites of {code}.",
    }]
    missing_idx = rng.randrange(len(groups))
    have = [g[0] for i, g in enumerate(groups) if i != missing_idx]
    missing = group_text(groups[missing_idx], names)
    q = (f"I have passed {join_list(have)}. Can I take {code} next semester?" if have
         else f"I haven't taken any courses yet. Can I register for {code}?")
    out.append({"type": "eligible_no", "course": code, "question": q,
                "answer": f"No; {code} requires {missing} first."})
    return out


def refusal_pairs(records, rng, n):
    out = []
    pool = list(records)
    for i in range(n):
        rec = rng.choice(pool)
        template = REFUSAL_TEMPLATES[i % len(REFUSAL_TEMPLATES)]
        out.append({"type": "refusal", "course": rec["code"],
                    "question": template.format(code=rec["code"], name=rec["name"]),
                    "answer": REFUSAL})
    return out


def test_courses(test_df, records):
    text = " ".join((test_df["question"] + " " + test_df["expected_answer"]).str.lower())
    questions = " ".join(test_df["question"].str.lower())
    hit = set()
    for r in records:
        code = r["code"].lower()
        if re.search(r"(?<![a-z0-9])" + re.escape(code) + r"(?![a-z0-9])", text) or r["name"].lower() in questions:
            hit.add(r["code"])
    return hit


def norm(s):
    return re.sub(r"\s+", " ", re.sub(r"[^\w\s]", " ", s.lower())).strip()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--refusals", type=int, default=60)
    args = ap.parse_args()
    cfg = load_config(args.config)
    p = cfg["paths"]
    rng = random.Random(cfg["seed"])
    records, names = load_courses(p["courses_csv"], p["raw_rows_csv"])
    tests = pd.read_csv(p["test_questions"], dtype=str).fillna("")
    excluded = test_courses(tests, records)
    usable = [r for r in records if r["code"] not in excluded]

    pairs = []
    for rec in usable:
        pairs.extend(course_pairs(rec, names, rng))
        pairs.extend(eligibility_pairs(rec, names, rng))
    pairs.extend(refusal_pairs(usable, rng, args.refusals))

    test_norm = [norm(q) for q in tests["question"]]
    seen, final = set(), []
    for pair in pairs:
        key = norm(pair["question"])
        if key in seen or key in test_norm:
            continue
        seen.add(key)
        final.append(pair)
    rng.shuffle(final)
    for i, pair in enumerate(final, 1):
        pair["id"] = f"sft_{i:04d}"
    n_val = round(len(final) * cfg["sft"]["val_fraction"])
    for i, pair in enumerate(final):
        pair["split"] = "val" if i < n_val else "train"

    assert 300 <= len(final) <= 1000, len(final)
    assert not any(pair["course"] in excluded for pair in final)

    with open(p["sft_pairs"], "w", encoding="utf-8") as f:
        for pair in final:
            f.write(json.dumps({k: pair[k] for k in ("id", "split", "type", "course", "question", "answer")},
                               ensure_ascii=False) + "\n")

    report = []
    for pair in final:
        scores = [difflib.SequenceMatcher(None, norm(pair["question"]), t).ratio() for t in test_norm]
        best = max(range(len(scores)), key=scores.__getitem__)
        report.append({"sft_id": pair["id"], "sft_question": pair["question"],
                       "closest_test_id": tests.iloc[best]["id"], "closest_test_question": tests.iloc[best]["question"],
                       "similarity": round(scores[best], 3)})
    report.sort(key=lambda r: -r["similarity"])
    os.makedirs(p["results_dir"], exist_ok=True)
    with open(os.path.join(p["results_dir"], "sft_overlap_report.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(report[0].keys()))
        w.writeheader()
        w.writerows(report)

    check = random.Random(cfg["seed"] + 1).sample(final, 50)
    with open(os.path.join(p["results_dir"], "sft_handcheck_50.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["id", "type", "course", "question", "answer", "correct_Y_N", "checked_by", "note"])
        for pair in check:
            w.writerow([pair["id"], pair["type"], pair["course"], pair["question"], pair["answer"], "", "", ""])

    counts = pd.Series([pair["type"] for pair in final]).value_counts().to_dict()
    summary = {"pairs": len(final), "train": len(final) - n_val, "val": n_val, "by_type": counts,
               "excluded_test_courses": sorted(excluded),
               "exact_test_question_copies": 0,
               "max_similarity_to_test": report[0]["similarity"]}
    with open(os.path.join(p["results_dir"], "sft_data_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
