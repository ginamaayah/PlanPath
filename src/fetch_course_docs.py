import argparse
import csv
import io
import os
import re
import sys
import time
import unicodedata

import pandas as pd
import requests
from bs4 import BeautifulSoup

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.common import load_config

SYLLABUS_URL = "https://services.just.edu.jo/CourseQA/Uploads/Syllabus/{line_no}.pdf"
DESCRIPTION_URL = ("https://services.just.edu.jo/CourseQA/WS/CourseInfo.aspx?"
                   "LineNo={line_no}&t=487f38e0-8c7e-4354-a21f-26b6779a50ba")
HEADERS = {"User-Agent": "Mozilla/5.0 (course project)"}
BLOCKED = ("request rejected", "the requested url was rejected")
MIN_CHARS = 200


def clean_text(text):
    text = unicodedata.normalize("NFKC", text)
    text = "".join(ch for ch in text if ch in "\n\t" or unicodedata.category(ch)[0] != "C")
    seen, lines = set(), []
    for line in text.splitlines():
        line = re.sub(r"\s+", " ", line).strip()
        if not line or re.fullmatch(r"(page\s*)?\d+(\s*(of|/)\s*\d+)?", line, re.I):
            continue
        if line in seen:
            continue
        seen.add(line)
        lines.append(line)
    return "\n".join(lines)


def pdf_text(content):
    from pypdf import PdfReader
    reader = PdfReader(io.BytesIO(content))
    return "\n".join((page.extract_text() or "") for page in reader.pages)


def html_text(content):
    soup = BeautifulSoup(content, "lxml")
    for tag in soup(["script", "style", "noscript"]):
        tag.decompose()
    return soup.get_text("\n")


def fetch(url, kind):
    try:
        resp = requests.get(url, headers=HEADERS, timeout=60)
    except requests.RequestException as exc:
        return None, f"error: {exc.__class__.__name__}"
    if resp.status_code != 200:
        return None, f"http {resp.status_code}"
    try:
        if kind == "syllabus" and resp.content[:4] == b"%PDF":
            raw = pdf_text(resp.content)
        else:
            raw = html_text(resp.content)
    except Exception as exc:
        return None, f"parse error: {exc.__class__.__name__}"
    text = clean_text(raw)
    if any(b in text.lower() for b in BLOCKED):
        return None, "blocked by site"
    if len(text) < MIN_CHARS:
        return None, f"too short ({len(text)} chars)"
    return text, "ok"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--delay", type=float, default=1.0)
    args = ap.parse_args()
    cfg = load_config(args.config)
    out = cfg["paths"]["course_docs_dir"]
    os.makedirs(out, exist_ok=True)
    courses = pd.read_csv(cfg["paths"]["courses_csv"], dtype=str)
    if args.limit:
        courses = courses.head(args.limit)
    report = []
    for _, r in courses.iterrows():
        for kind, url in (("description", DESCRIPTION_URL), ("syllabus", SYLLABUS_URL)):
            link = url.format(line_no=r["line_no"])
            text, status = fetch(link, kind)
            if text:
                with open(os.path.join(out, f"{r['line_no']}_{kind}.txt"), "w", encoding="utf-8") as f:
                    f.write(text)
            report.append({"line_no": r["line_no"], "code": r["code"], "kind": kind,
                           "status": status, "characters": len(text) if text else 0, "url": link})
            print(f"{r['code']:<12} {kind:<12} {status}")
            time.sleep(args.delay)
    with open(os.path.join(cfg["paths"]["cpt_dir"], "course_docs_report.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(report[0].keys()))
        w.writeheader()
        w.writerows(report)
    ok = sum(1 for x in report if x["status"] == "ok")
    print(f"\nDownloaded {ok} of {len(report)} documents into {out}")


if __name__ == "__main__":
    main()
