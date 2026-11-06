import csv
import json
import math
import os

import torch
from transformers import TrainerCallback, TrainingArguments


class EpochEarlyStopping(TrainerCallback):
    def __init__(self, patience):
        self.patience = patience
        self.pending = False
        self.last_eval = None
        self.epoch_evals = []
        self.best = None
        self.bad_epochs = 0

    def on_epoch_end(self, args, state, control, **kwargs):
        if self.last_eval and self.last_eval[0] == state.global_step:
            self._record(state, self.last_eval[1], control)
        else:
            control.should_evaluate = True
            self.pending = True
        return control

    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        if metrics is None:
            return control
        self.last_eval = (state.global_step, metrics["eval_loss"])
        if self.pending:
            self.pending = False
            self._record(state, metrics["eval_loss"], control)
        return control

    def _record(self, state, loss, control):
        self.epoch_evals.append({"epoch": round(state.epoch), "step": state.global_step, "eval_loss": loss})
        if self.best is None or loss < self.best:
            self.best = loss
            self.bad_epochs = 0
        else:
            self.bad_epochs += 1
            if self.bad_epochs >= self.patience:
                control.should_training_stop = True

    def best_epoch(self):
        return min(self.epoch_evals, key=lambda e: e["eval_loss"])


def training_arguments(cfg_stage, output_dir, seed, steps_per_epoch, eval_steps, save_strategy,
                       save_steps=None, load_best=False, save_total_limit=None):
    total = steps_per_epoch * cfg_stage["epochs"]
    cuda = torch.cuda.is_available()
    bf16 = cuda and torch.cuda.is_bf16_supported()
    return TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=cfg_stage["epochs"],
        learning_rate=float(cfg_stage["learning_rate"]),
        per_device_train_batch_size=cfg_stage["batch_size"],
        per_device_eval_batch_size=cfg_stage["batch_size"],
        gradient_accumulation_steps=cfg_stage["gradient_accumulation_steps"],
        weight_decay=cfg_stage["weight_decay"],
        warmup_steps=int(total * cfg_stage["warmup_ratio"]),
        lr_scheduler_type="linear",
        eval_strategy="steps",
        eval_steps=eval_steps,
        logging_strategy="steps",
        logging_steps=max(1, eval_steps // 2),
        logging_first_step=True,
        save_strategy=save_strategy,
        save_steps=save_steps or eval_steps,
        save_total_limit=save_total_limit,
        load_best_model_at_end=load_best,
        metric_for_best_model="eval_loss" if load_best else None,
        greater_is_better=False if load_best else None,
        save_only_model=True,
        report_to="none",
        seed=seed,
        data_seed=seed,
        bf16=bf16,
        fp16=cuda and not bf16,
        dataloader_num_workers=0,
        remove_unused_columns=False,
    )


def steps_per_epoch(n_examples, cfg_stage):
    return max(1, math.ceil(n_examples / (cfg_stage["batch_size"] * cfg_stage["gradient_accumulation_steps"])))


def save_log_history(log_history, results_dir, prefix):
    os.makedirs(results_dir, exist_ok=True)
    with open(os.path.join(results_dir, f"{prefix}_log_history.json"), "w") as f:
        json.dump(log_history, f, indent=2)
    rows = []
    for e in log_history:
        if "loss" in e:
            rows.append({"step": e["step"], "epoch": e.get("epoch"), "split": "train", "loss": e["loss"]})
        if "eval_loss" in e:
            rows.append({"step": e["step"], "epoch": e.get("epoch"), "split": "validation", "loss": e["eval_loss"]})
    with open(os.path.join(results_dir, f"{prefix}_loss.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["step", "epoch", "split", "loss"])
        w.writeheader()
        w.writerows(rows)
    plot_losses(rows, os.path.join(results_dir, f"{prefix}_loss_curve.png"), prefix)
    return rows


def plot_losses(rows, path, prefix):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    titles = {"cpt": "Continued pretraining", "sft": "Supervised fine-tuning"}
    fig, ax = plt.subplots(figsize=(6, 3.6))
    for split, style in (("train", "-"), ("validation", "o-")):
        pts = [(r["step"], r["loss"]) for r in rows if r["split"] == split]
        if pts:
            ax.plot([p[0] for p in pts], [p[1] for p in pts], style, label=f"{split} loss", markersize=4)
    ax.set_xlabel("training step")
    ax.set_ylabel("cross-entropy loss")
    ax.set_title(titles.get(prefix, prefix))
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
