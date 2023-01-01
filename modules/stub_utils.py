# -*- coding: utf-8 -*-
# utils for faking data
import random
import string

from transformers import AutoTokenizer, DataCollatorForSeq2Seq

dummy_input = "Batches of inputs. Loreum Ipsum"
dummy_outputs = (
    "Batches of outputs, I am sorrt I am just a bot I have no idea what you are taking"
)


def encode_batch(text, target, tokenizer):
    random_text = "".join(random.choice(string.ascii_letters) for x in range(10))
    input_encodings = tokenizer(
        text + random_text * int(random.random() * 100), max_length=301, truncation=True
    )

    with tokenizer.as_target_tokenizer():
        target_encodings = tokenizer(
            target + "<random>" * int(random.random() * 100),
            max_length=401,
            truncation=True,
        )
    return {
        "input_ids": input_encodings["input_ids"],
        "attention_mask": input_encodings["attention_mask"],
        "labels": target_encodings["input_ids"],
    }


def get_batch_of_inputs(tokenizer):
    entry = encode_batch(dummy_input, dummy_outputs, tokenizer)
    return entry


def gen_batch(tokenizer, model, batch_size=64):
    collate_fn = DataCollatorForSeq2Seq(tokenizer, model, max_length=402)
    while True:
        batch = []
        for _ in range(batch_size):
            batch.append(get_batch_of_inputs(tokenizer))
        yield collate_fn(batch)


if __name__ == "__main__":
    """
    python -m modules.stub_utils
    """
    from transformers import AutoModelForSeq2SeqLM

    tokenizer = AutoTokenizer.from_pretrained("google/mt5-small")

    model = AutoModelForSeq2SeqLM.from_pretrained("google/mt5-small")
    generator = gen_batch(tokenizer, model)
    for batch in generator:  # generates a random size of
        print(batch["input_ids"].shape)
