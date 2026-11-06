import argparse
import json
import os
import shutil

from transformers import EarlyStoppingCallback, Trainer

from src.chat_template import END, SPECIAL_TOKENS, add_chat_tokens
from src.common import checkpoint_dir, load_config, model_paths, set_seed
from src.data_utils import PadCollator, SFTDataset, read_jsonl
from src.models import generate_answer, load_model
from src.training_utils import save_log_history, steps_per_epoch, training_arguments


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--fraction", type=float, default=1.0)
    ap.add_argument("--epochs", type=int, default=None)
    args = ap.parse_args()
    cfg = load_config(args.config)
    c = cfg["sft"]
    if args.epochs:
        c["epochs"] = args.epochs
    set_seed(cfg["seed"])
    results = cfg["paths"]["results_dir"]
    out_dir = checkpoint_dir(cfg, "sft")
    if os.path.exists(out_dir):
        shutil.rmtree(out_dir)
    os.makedirs(results, exist_ok=True)

    pairs = read_jsonl(cfg["paths"]["sft_pairs"])
    train_pairs = [p for p in pairs if p["split"] == "train"]
    val_pairs = [p for p in pairs if p["split"] == "val"]
    if args.fraction < 1.0:
        train_pairs = train_pairs[:max(1, int(len(train_pairs) * args.fraction))]
    print(f"SFT pairs: {len(train_pairs)} train, {len(val_pairs)} validation")

    model, tokenizer = load_model(model_paths(cfg)["cpt"])
    added = add_chat_tokens(tokenizer, model)
    print(f"added chat tokens: {added} -> {SPECIAL_TOKENS}")
    train_ds = SFTDataset(tokenizer, train_pairs, c["max_length"])
    val_ds = SFTDataset(tokenizer, val_pairs, c["max_length"])
    answer_tokens = sum(sum(1 for t in x["labels"] if t != -100) for x in train_ds.items)
    all_tokens = sum(len(x["input_ids"]) for x in train_ds.items)
    print(f"loss is computed on {answer_tokens} of {all_tokens} training tokens (answers only)")

    spe = steps_per_epoch(len(train_ds), c)
    eval_steps = max(1, spe // 2)
    targs = training_arguments(c, out_dir, cfg["seed"], spe, eval_steps, save_strategy="steps",
                               save_steps=eval_steps, load_best=True, save_total_limit=2)
    trainer = Trainer(
        model=model,
        args=targs,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        data_collator=PadCollator(tokenizer.pad_token_id),
        callbacks=[EarlyStoppingCallback(early_stopping_patience=2)],
    )
    trainer.train()
    best_loss = trainer.state.best_metric

    final_dir = os.path.join(out_dir, "final")
    if os.path.exists(final_dir):
        shutil.rmtree(final_dir)
    trainer.model.save_pretrained(final_dir)
    tokenizer.save_pretrained(final_dir)
    save_log_history(trainer.state.log_history, results, "sft")

    model, tokenizer = load_model(final_dir)
    samples = []
    for p in val_pairs[:20]:
        answer, _ = generate_answer(model, tokenizer, p["question"], cfg["generation"]["max_new_tokens"])
        samples.append((p, answer))
    with open(os.path.join(results, "sft_val_samples.md"), "w", encoding="utf-8") as f:
        f.write("# SFT model on 20 validation questions (greedy)\n\n")
        for p, a in samples:
            f.write(f"**Q:** {p['question']}\n\n**Expected:** {p['answer']}\n\n**Model:** {a}\n\n---\n\n")
    summary = {
        "start_model": model_paths(cfg)["cpt"], "chat_template": f"<|user|>\\n{{question}}\\n<|assistant|>\\n{{answer}}{END}",
        "train_pairs": len(train_pairs), "val_pairs": len(val_pairs),
        "answer_tokens": answer_tokens, "all_tokens": all_tokens,
        "epochs_planned": c["epochs"], "learning_rate": c["learning_rate"],
        "best_validation_loss": best_loss, "best_checkpoint": trainer.state.best_model_checkpoint,
        "final_checkpoint": final_dir,
    }
    with open(os.path.join(results, "sft_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
