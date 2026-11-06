USER = "<|user|>"
ASSISTANT = "<|assistant|>"
END = "<|end|>"
SPECIAL_TOKENS = [USER, ASSISTANT, END]


def format_prompt(question):
    return f"{USER}\n{question.strip()}\n{ASSISTANT}\n"


def format_answer(answer):
    return f"{answer.strip()}{END}"


def add_chat_tokens(tokenizer, model):
    added = tokenizer.add_tokens(SPECIAL_TOKENS, special_tokens=True)
    if added:
        model.resize_token_embeddings(len(tokenizer))
    return added


def clean_generation(text):
    for stop in (END, USER, "<|endoftext|>"):
        if stop in text:
            text = text.split(stop)[0]
    return text.strip()
