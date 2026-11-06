import json
import math

import torch
from torch.utils.data import Dataset

from src.chat_template import format_answer, format_prompt


def read_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def pack_documents(tokenizer, texts, block_size):
    ids = []
    for text in texts:
        ids.extend(tokenizer(text)["input_ids"] + [tokenizer.eos_token_id])
    return [ids[i:i + block_size] for i in range(0, len(ids), block_size)]


class BlockDataset(Dataset):
    def __init__(self, blocks):
        self.blocks = blocks

    def __len__(self):
        return len(self.blocks)

    def __getitem__(self, i):
        ids = self.blocks[i]
        return {"input_ids": ids, "labels": list(ids)}


class SFTDataset(Dataset):
    def __init__(self, tokenizer, pairs, max_length):
        self.items = [encode_pair(tokenizer, p["question"], p["answer"], max_length) for p in pairs]

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        return self.items[i]


def encode_pair(tokenizer, question, answer, max_length):
    prompt_ids = tokenizer(format_prompt(question))["input_ids"]
    answer_ids = tokenizer(format_answer(answer))["input_ids"]
    input_ids = (prompt_ids + answer_ids)[:max_length]
    labels = ([-100] * len(prompt_ids) + answer_ids)[:max_length]
    return {"input_ids": input_ids, "labels": labels}


class PadCollator:
    def __init__(self, pad_token_id):
        self.pad_token_id = pad_token_id

    def __call__(self, batch):
        width = max(len(x["input_ids"]) for x in batch)
        input_ids, labels, mask = [], [], []
        for x in batch:
            n = len(x["input_ids"])
            pad = width - n
            input_ids.append(x["input_ids"] + [self.pad_token_id] * pad)
            labels.append(x["labels"] + [-100] * pad)
            mask.append([1] * n + [0] * pad)
        return {"input_ids": torch.tensor(input_ids), "labels": torch.tensor(labels),
                "attention_mask": torch.tensor(mask)}


@torch.no_grad()
def perplexity(model, tokenizer, texts, block_size, batch_size=4):
    model.eval()
    blocks = pack_documents(tokenizer, texts, block_size)
    collate = PadCollator(tokenizer.pad_token_id)
    total_nll, total_tokens = 0.0, 0
    for i in range(0, len(blocks), batch_size):
        batch = collate([{"input_ids": b, "labels": list(b)} for b in blocks[i:i + batch_size]])
        batch = {k: v.to(model.device) for k, v in batch.items()}
        logits = model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"]).logits
        shift_logits = logits[:, :-1].float()
        shift_labels = batch["labels"][:, 1:]
        loss = torch.nn.functional.cross_entropy(
            shift_logits.reshape(-1, shift_logits.size(-1)), shift_labels.reshape(-1),
            ignore_index=-100, reduction="sum")
        total_nll += loss.item()
        total_tokens += int((shift_labels != -100).sum())
    return math.exp(total_nll / total_tokens), total_tokens
