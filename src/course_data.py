import pandas as pd

PLAN_NAME = "JUST B.Sc. Computer Information Systems (CIS) study plan (plan no. 766)"

SECTION_NAMES = {
    "University Compulsory Req.": "University Compulsory Requirements",
    "University Elective Req.": "University Elective Requirements",
    "Faculty Compulsory Req.": "Faculty Compulsory Requirements",
    "Department Compulsory Req.": "Department Compulsory Requirements",
    "Department Elective Req.": "Department Elective Requirements",
}

TEACHING = {
    "On Campus": "on campus",
    "Electronic Course": "as an electronic (online) course",
    "Blended": "in blended mode",
}

RELATION_COLUMNS = ["prereq_pass", "prereq_study", "prereq_or_concurrent", "concurrent"]


def title(name):
    small = {"and", "of", "in", "for", "the", "to", "a", "an", "&"}
    words = " ".join(str(name).split()).split(" ")
    out = []
    for i, w in enumerate(words):
        if i > 0 and w.lower() in small:
            out.append(w.lower())
        elif w.lower() in ("ai", "it", "gis", "erp"):
            out.append(w.upper())
        else:
            out.append(w[:1].upper() + w[1:])
    return " ".join(out)


def load_courses(courses_csv, raw_rows_csv):
    courses = pd.read_csv(courses_csv, dtype=str).fillna("")
    raw = pd.read_csv(raw_rows_csv, dtype=str).fillna("")
    names = {}
    for _, r in raw.iterrows():
        names.setdefault(r["code"], title(r["course"]))
    for _, r in courses.iterrows():
        names[r["code"]] = title(r["course"])
    records = []
    for _, r in courses.iterrows():
        rec = {
            "line_no": r["line_no"],
            "code": r["code"],
            "name": title(r["course"]),
            "hours": int(r["hours"]),
            "section": SECTION_NAMES[r["section"]],
            "section_hours": int(r["section_hours"]),
            "compulsory": "Compulsory" in r["section"],
            "teaching": TEACHING.get(r["teaching_method"], r["teaching_method"].lower()),
            "equivalents": split_codes(r["equivalents"]),
        }
        for col in RELATION_COLUMNS:
            rec[col] = parse_groups(r[col])
        records.append(rec)
    return records, names


def split_codes(value):
    out = []
    for group in parse_groups(value):
        out.extend(group)
    return list(dict.fromkeys(out))


def parse_groups(value):
    value = str(value).strip()
    if not value:
        return []
    return [[c.strip() for c in g.split("/") if c.strip()] for g in value.split(";") if g.strip()]


def label(code, names):
    name = names.get(code)
    return f"{code} ({name})" if name else code


def group_text(group, names):
    if len(group) == 1:
        return label(group[0], names)
    name = names.get(group[0])
    codes = " or ".join(group)
    return f"{name} ({codes})" if name else codes


def join_list(items):
    if len(items) <= 1:
        return "".join(items)
    return ", ".join(items[:-1]) + " and " + items[-1]


def has_requirements(rec):
    return any(rec[c] for c in RELATION_COLUMNS)
