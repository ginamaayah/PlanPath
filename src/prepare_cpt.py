import argparse
import json
import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.common import load_config
from src.course_data import (PLAN_NAME, group_text, join_list, label,
                             load_courses)


def hours_text(hours):
    return f"{hours} credit hour" + ("" if hours == 1 else "s")


def course_document(rec, names):
    code, name = rec["code"], rec["name"]
    hours = hours_text(rec["hours"])
    kind = "a compulsory" if rec["compulsory"] else "an elective"
    lines = [
        f"{code} {name}",
        f"{code} ({name}) is a course of {hours} in the {rec['section']} of the {PLAN_NAME}.",
        f"It is {kind} course in the plan and it is taught {rec['teaching']}.",
    ]
    if rec["prereq_study"]:
        lines.append(f"Before taking {code}, students must have studied "
                     f"{join_list([group_text(g, names) for g in rec['prereq_study']])}.")
    if rec["prereq_pass"]:
        lines.append(f"Before taking {code}, students must have passed "
                     f"{join_list([group_text(g, names) for g in rec['prereq_pass']])}.")
    if rec["prereq_or_concurrent"]:
        lines.append(f"Students must take {join_list([group_text(g, names) for g in rec['prereq_or_concurrent']])} "
                     f"before {code} or in the same semester.")
    if rec["concurrent"]:
        lines.append(f"{code} must be taken in the same semester as "
                     f"{join_list([group_text(g, names) for g in rec['concurrent']])}.")
    if not any(rec[c] for c in ("prereq_study", "prereq_pass", "prereq_or_concurrent", "concurrent")):
        lines.append(f"{code} has no prerequisite.")
    eq = [c for c in rec["equivalents"] if c != code]
    if eq:
        verb = "is" if len(eq) == 1 else "are"
        lines.append(f"{join_list([label(c, names) for c in eq])} {verb} listed as equivalent to {code}.")
    return "\n".join(lines)


def section_documents(records):
    docs = []
    sections = {}
    for rec in records:
        sections.setdefault(rec["section"], {"hours": rec["section_hours"], "courses": []})
        sections[rec["section"]]["courses"].append(rec)
    for section, info in sections.items():
        items = [f"{r['code']} {r['name']} ({hours_text(r['hours'])})" for r in info["courses"]]
        if "Compulsory" in section:
            head = (f"The {section} of the {PLAN_NAME} total {info['hours']} credit hours. "
                    f"Students must take all of these courses:")
        else:
            head = (f"Students must complete {info['hours']} credit hours of {section} in the {PLAN_NAME}. "
                    f"They choose their courses from this list:")
        docs.append({"doc_id": "section_" + section.lower().replace(" ", "_"),
                     "source": "study_plan_section", "text": head + "\n" + "\n".join(items)})
    total = sum(info["hours"] for info in sections.values())
    parts = [f"{info['hours']} credit hours of {s}" for s, info in sections.items()]
    overview = (f"The {PLAN_NAME} has {len(sections)} requirement groups: {join_list(parts)}. "
                f"In total the plan has {total} credit hours.")
    docs.append({"doc_id": "plan_overview", "source": "study_plan_overview", "text": overview})
    return docs


def read_course_docs(folder, line_no):
    out = []
    for kind, heading in (("description", "Course description"), ("syllabus", "Syllabus")):
        path = os.path.join(folder, f"{line_no}_{kind}.txt")
        if os.path.exists(path):
            text = open(path, encoding="utf-8").read().strip()
            if text:
                out.append((kind, heading, text))
    return out


def build_documents(cfg):
    p = cfg["paths"]
    records, names = load_courses(p["courses_csv"], p["raw_rows_csv"])
    docs = []
    used_sources = set()
    for rec in records:
        text = course_document(rec, names)
        sources = ["study_plan"]
        if cfg["cpt"].get("use_course_docs", True):
            for kind, heading, extra in read_course_docs(p["course_docs_dir"], rec["line_no"]):
                text += f"\n\n{heading} of {rec['code']} {rec['name']}:\n{extra}"
                sources.append(kind)
                used_sources.add(kind)
        docs.append({"doc_id": rec["code"], "source": "+".join(sources), "text": text})
    docs.extend(section_documents(records))
    return docs, used_sources


def count_tokens(cfg, docs):
    try:
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained(cfg["model"]["base"])
    except Exception as exc:
        print(f"Token count skipped ({exc.__class__.__name__}).")
        return None
    return sum(len(tok(d["text"])["input_ids"]) + 1 for d in docs)


def write_jsonl(path, rows):
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.yaml")
    args = ap.parse_args()
    cfg = load_config(args.config)
    docs, used = build_documents(cfg)
    rng = random.Random(cfg["seed"])
    order = list(range(len(docs)))
    rng.shuffle(order)
    n_held = max(1, round(len(docs) * cfg["cpt"]["heldout_fraction"]))
    held_ids = set(order[:n_held])
    train = [d for i, d in enumerate(docs) if i not in held_ids]
    held = [d for i, d in enumerate(docs) if i in held_ids]
    out = cfg["paths"]["cpt_dir"]
    os.makedirs(out, exist_ok=True)
    write_jsonl(os.path.join(out, "train.jsonl"), train)
    write_jsonl(os.path.join(out, "heldout.jsonl"), held)
    stats = {
        "documents": len(docs), "train_documents": len(train), "heldout_documents": len(held),
        "heldout_doc_ids": [d["doc_id"] for d in held],
        "course_docs_used": sorted(used),
        "train_tokens": count_tokens(cfg, train), "heldout_tokens": count_tokens(cfg, held),
        "train_characters": sum(len(d["text"]) for d in train),
        "heldout_characters": sum(len(d["text"]) for d in held),
    }
    with open(os.path.join(out, "corpus_stats.json"), "w") as f:
        json.dump(stats, f, indent=2)
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
