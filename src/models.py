import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.chat_template import END, clean_generation, format_prompt


def get_device():
    return "cuda" if torch.cuda.is_available() else "cpu"


def load_model(name_or_path, device=None):
    device = device or get_device()
    tokenizer = AutoTokenizer.from_pretrained(name_or_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(name_or_path)
    model.to(device)
    model.eval()
    return model, tokenizer


def stop_token_ids(tokenizer):
    ids = [tokenizer.eos_token_id]
    if END in tokenizer.get_vocab():
        ids.append(tokenizer.convert_tokens_to_ids(END))
    return ids


@torch.no_grad()
def generate_answer(model, tokenizer, question, max_new_tokens):
    prompt = format_prompt(question)
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    if model.device.type == "cuda":
        torch.cuda.synchronize()
    start = time.perf_counter()
    output = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        eos_token_id=stop_token_ids(tokenizer),
        pad_token_id=tokenizer.pad_token_id,
    )
    if model.device.type == "cuda":
        torch.cuda.synchronize()
    latency = time.perf_counter() - start
    new_tokens = output[0][inputs["input_ids"].shape[1]:]
    text = tokenizer.decode(new_tokens, skip_special_tokens=False)
    return clean_generation(text), latency


@torch.no_grad()
def complete(model, tokenizer, prompt, max_new_tokens):
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    output = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        pad_token_id=tokenizer.pad_token_id,
    )
    return tokenizer.decode(output[0], skip_special_tokens=True)
