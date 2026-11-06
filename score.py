import argparse
import gc
import json
import os
import platform

import pandas as pd
import torch
import transformers

from src.common import load_config, model_paths, set_seed
from src.data_utils import perplexity, read_jsonl
from src.models import generate_answer, load_model
from src.scoring import ANSWER_KEYS, auto_score

MODELS = ["base", "cpt", "sft"]
LABELS = {"base": "Small model, base", "cpt": "+ continued pretraining", "sft": "+ SFT"}
GRADES = {"0", "0.5", "1", "0.0", "1.0"}


def load_tests(cfg, limit=None):
    tests = pd.read_csv(cfg["paths"]["test_questions"], dtype=str).fillna("")
    keys = pd.DataFrame([{"id": k, "scoring": v[0], "key": v[1], "options": v[2]} for k, v in ANSWER_KEYS.items()])
    tests = tests.merge(keys, on="id", how="left")
    return tests.head(limit) if limit else tests


def run_model(cfg, name, path, tests, held_texts, results_dir):
    model, tokenizer = load_model(path)
    rows = []
    for _, t in tests.iterrows():
        answer, latency = generate_answer(model, tokenizer, t["question"], cfg["generation"]["max_new_tokens"])
        score, note, extracted = auto_score(answer, t["scoring"], t["key"], t["options"])
        rows.append({"id": t["id"], "question": t["question"], "expected_answer": t["expected_answer"],
                     "model_answer": answer, "scoring": t["scoring"], "key": t["key"],
                     "extracted": extracted, "score": "" if score is None else score,
                     "note": note, "latency_s": round(latency, 3)})
        print(f"[{name}] Q{t['id']}: {answer[:80]!r} -> {note}")
    ppl = None
    if held_texts:
        ppl, _ = perplexity(model, tokenizer, held_texts, cfg["cpt"]["block_size"])
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(results_dir, f"results_{name}.csv"), index=False, encoding="utf-8")
    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return df, ppl


def write_grading_sheet(results_dir, frames):
    rows = []
    for name, df in frames.items():
        for _, r in df[df["scoring"].isin(["hand"])].iterrows():
            rows.append({"model": name, "id": r["id"], "question": r["question"],
                         "expected_answer": r["expected_answer"], "model_answer": r["model_answer"],
                         "grader_1": "", "grader_2": "", "agreed": "", "note": r["note"]})
    path = os.path.join(results_dir, "grading_sheet.csv")
    pd.DataFrame(rows).to_csv(path, index=False, encoding="utf-8")
    return path


def generate(cfg, models, limit, skip_ppl):
    results_dir = cfg["paths"]["results_dir"]
    os.makedirs(results_dir, exist_ok=True)
    tests = load_tests(cfg, limit)
    held_path = os.path.join(cfg["paths"]["cpt_dir"], "heldout.jsonl")
    held_texts = [] if skip_ppl else [d["text"] for d in read_jsonl(held_path)]
    paths = model_paths(cfg)
    frames, ppls = {}, {}
    for name in models:
        frames[name], ppls[name] = run_model(cfg, name, paths[name], tests, held_texts, results_dir)
    meta = {"models": {m: paths[m] for m in models}, "questions": len(tests),
            "decoding": "greedy (do_sample=False)", "max_new_tokens": cfg["generation"]["max_new_tokens"],
            "heldout_perplexity": ppls, "device": "cuda" if torch.cuda.is_available() else "cpu",
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            "transformers": transformers.__version__, "torch": torch.__version__, "python": platform.python_version()}
    with open(os.path.join(results_dir, "run_meta.json"), "w") as f:
        json.dump(meta, f, indent=2)
    sheet = write_grading_sheet(results_dir, frames)
    print("\nAutomatic scores (hand-graded questions not counted yet):")
    for name, df in frames.items():
        auto = pd.to_numeric(df["score"], errors="coerce").dropna()
        print(f"  {name}: {auto.sum():.1f} / {len(auto)} automatic questions correct")
    print(f"\nNext: two team members fill grader_1 and grader_2 (1, 0.5 or 0) in {sheet}, "
          f"settle disagreements in 'agreed', then run: python score.py --finalize")


def read_json(path):
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return {}


def finalize(cfg):
    results_dir = cfg["paths"]["results_dir"]
    sheet = pd.read_csv(os.path.join(results_dir, "grading_sheet.csv"), dtype=str).fillna("")
    problems = []
    agree = 0
    finals = {}
    for _, r in sheet.iterrows():
        g1, g2, ag = r["grader_1"].strip(), r["grader_2"].strip(), r["agreed"].strip()
        if g1 not in GRADES or g2 not in GRADES:
            problems.append(f"{r['model']} Q{r['id']}: grader_1/grader_2 must be 1, 0.5 or 0")
            continue
        if float(g1) == float(g2):
            agree += 1
            finals[(r["model"], r["id"])] = float(g1)
        elif ag in GRADES:
            finals[(r["model"], r["id"])] = float(ag)
        else:
            problems.append(f"{r['model']} Q{r['id']}: graders disagree, fill 'agreed'")
    if problems:
        raise SystemExit("Grading sheet not complete:\n  " + "\n  ".join(problems))
    agreement = agree / len(sheet) if len(sheet) else None

    meta = read_json(os.path.join(results_dir, "run_meta.json"))
    cpt = read_json(os.path.join(results_dir, "cpt_summary.json"))
    sft = read_json(os.path.join(results_dir, "sft_summary.json"))
    notes = {
        "base": "GPT-2 small (124M parameters)" if cfg["model"]["base"] == "gpt2" else cfg["model"]["base"],
        "cpt": (f"{cpt.get('epochs_run', '?')} passes (best: pass {cpt.get('best_epoch', '?')}), "
                f"learning rate {cfg['cpt']['learning_rate']}"),
        "sft": f"{sft.get('train_pairs', '?')} training pairs + {sft.get('val_pairs', '?')} validation pairs",
    }
    summary = {"grader_initial_agreement": agreement, "hand_graded_items": len(sheet), "models": {}}
    lines = ["# Scoreboard", "",
             "Same 30 Phase 1 test questions for every version; greedy decoding, "
             f"max {meta.get('max_new_tokens', cfg['generation']['max_new_tokens'])} new tokens.", "",
             "| Version | Held-out perplexity | Accuracy | Refusal accuracy | Avg. latency (s) | Notes |",
             "|---|---|---|---|---|---|"]
    for name in MODELS:
        path = os.path.join(results_dir, f"results_{name}.csv")
        if not os.path.exists(path):
            continue
        df = pd.read_csv(path, dtype=str).fillna("")
        for i, r in df.iterrows():
            if r["scoring"] == "hand":
                df.at[i, "score"] = str(finals[(name, r["id"])])
        df.to_csv(path, index=False, encoding="utf-8")
        scores = pd.to_numeric(df["score"])
        refuse = df["scoring"] == "refuse"
        acc = scores.mean() * 100
        racc = scores[refuse].mean() * 100 if refuse.any() else float("nan")
        lat = pd.to_numeric(df["latency_s"]).mean()
        ppl = meta.get("heldout_perplexity", {}).get(name)
        summary["models"][name] = {"accuracy": round(acc, 1), "refusal_accuracy": round(racc, 1),
                                   "avg_latency_s": round(lat, 3), "heldout_perplexity": ppl,
                                   "points": float(scores.sum()), "questions": len(df)}
        ppl_txt = f"{ppl:.2f}" if ppl else "n/a"
        lines.append(f"| {LABELS[name]} | {ppl_txt} | {acc:.1f}% | {racc:.0f}% | {lat:.2f} | {notes[name]} |")
    if agreement is not None:
        lines += ["", f"Hand-graded answers: {len(sheet)}; the two graders agreed at first on "
                      f"{agree} of {len(sheet)} ({agreement * 100:.0f}%)."]
    with open("SCOREBOARD.md", "w") as f:
        f.write("\n".join(lines) + "\n")
    with open(os.path.join(results_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print("\n".join(lines))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--models", nargs="+", default=MODELS, choices=MODELS)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--skip-perplexity", action="store_true")
    ap.add_argument("--finalize", action="store_true")
    args = ap.parse_args()
    cfg = load_config(args.config)
    set_seed(cfg["seed"])
    if args.finalize:
        finalize(cfg)
    else:
        generate(cfg, args.models, args.limit, args.skip_perplexity)


if __name__ == "__main__":
    main()
