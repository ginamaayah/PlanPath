import argparse
import json
import os
import shutil

from transformers import Trainer

from src.common import checkpoint_dir, load_config, set_seed
from src.data_utils import BlockDataset, PadCollator, pack_documents, perplexity, read_jsonl
from src.models import complete, load_model
from src.training_utils import (EpochEarlyStopping, save_log_history, steps_per_epoch,
                                training_arguments)


def completions(model, tokenizer, prompts, max_new_tokens):
    return [{"prompt": p, "completion": complete(model, tokenizer, p, max_new_tokens)[len(p):].strip()}
            for p in prompts]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--fraction", type=float, default=1.0)
    ap.add_argument("--epochs", type=int, default=None)
    args = ap.parse_args()
    cfg = load_config(args.config)
    c = cfg["cpt"]
    if args.epochs:
        c["epochs"] = args.epochs
    set_seed(cfg["seed"])
    results = cfg["paths"]["results_dir"]
    out_dir = checkpoint_dir(cfg, "cpt")
    if os.path.exists(out_dir):
        shutil.rmtree(out_dir)
    os.makedirs(results, exist_ok=True)

    train_docs = read_jsonl(os.path.join(cfg["paths"]["cpt_dir"], "train.jsonl"))
    held_docs = read_jsonl(os.path.join(cfg["paths"]["cpt_dir"], "heldout.jsonl"))
    if args.fraction < 1.0:
        train_docs = train_docs[:max(1, int(len(train_docs) * args.fraction))]
    train_texts = [d["text"] for d in train_docs]
    held_texts = [d["text"] for d in held_docs]

    model, tokenizer = load_model(cfg["model"]["base"])
    train_blocks = pack_documents(tokenizer, train_texts, c["block_size"])
    held_blocks = pack_documents(tokenizer, held_texts, c["block_size"])
    print(f"train: {len(train_docs)} docs, {sum(map(len, train_blocks))} tokens, {len(train_blocks)} blocks")
    print(f"held-out: {len(held_docs)} docs, {sum(map(len, held_blocks))} tokens, {len(held_blocks)} blocks")

    ppl_before, n_tokens = perplexity(model, tokenizer, held_texts, c["block_size"])
    print(f"held-out perplexity before: {ppl_before:.2f}")
    before = completions(model, tokenizer, c["prompts"], c["completion_max_new_tokens"])

    spe = steps_per_epoch(len(train_blocks), c)
    eval_steps = max(1, spe // c["evals_per_epoch"])
    targs = training_arguments(c, out_dir, cfg["seed"], spe, eval_steps, save_strategy="epoch")
    stopper = EpochEarlyStopping(c["early_stopping_patience"])
    trainer = Trainer(
        model=model,
        args=targs,
        train_dataset=BlockDataset(train_blocks),
        eval_dataset=BlockDataset(held_blocks),
        data_collator=PadCollator(tokenizer.pad_token_id),
        callbacks=[stopper],
    )
    trainer.train()

    best = stopper.best_epoch()
    best_dir = os.path.join(out_dir, f"checkpoint-{best['step']}")
    print(f"best epoch: {best['epoch']} (step {best['step']}, validation loss {best['eval_loss']:.4f})")
    final_dir = os.path.join(out_dir, "final")
    if os.path.exists(final_dir):
        shutil.rmtree(final_dir)
    best_model, _ = load_model(best_dir)
    best_model.save_pretrained(final_dir)
    tokenizer.save_pretrained(final_dir)

    ppl_after, _ = perplexity(best_model, tokenizer, held_texts, c["block_size"])
    print(f"held-out perplexity after: {ppl_after:.2f}")
    after = completions(best_model, tokenizer, c["prompts"], c["completion_max_new_tokens"])

    save_log_history(trainer.state.log_history, results, "cpt")
    summary = {
        "model": cfg["model"]["base"], "train_docs": len(train_docs), "heldout_docs": len(held_docs),
        "train_tokens": sum(map(len, train_blocks)), "heldout_tokens": sum(map(len, held_blocks)),
        "block_size": c["block_size"], "learning_rate": c["learning_rate"],
        "epochs_planned": c["epochs"], "epochs_run": len(stopper.epoch_evals),
        "epoch_validation_loss": stopper.epoch_evals, "best_epoch": best["epoch"],
        "heldout_perplexity_before": round(ppl_before, 3), "heldout_perplexity_after": round(ppl_after, 3),
        "final_checkpoint": final_dir,
    }
    with open(os.path.join(results, "cpt_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    with open(os.path.join(results, "cpt_completions.md"), "w", encoding="utf-8") as f:
        f.write("# Completions before and after continued pretraining (greedy)\n\n")
        for b, a in zip(before, after):
            f.write(f"## Prompt: {b['prompt']}\n\n**Before:** {b['completion']}\n\n**After:** {a['completion']}\n\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
