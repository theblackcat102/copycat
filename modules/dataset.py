# -*- coding: utf-8 -*-
import glob
import os
import re
from functools import lru_cache

import ujson as json
from bs4 import BeautifulSoup
from datasets import load_dataset
from markdownify import markdownify
from torch.utils.data import Dataset
from tqdm import tqdm
from transformers import AutoTokenizer, DataCollatorForSeq2Seq

invalid_code = re.compile(r"<<[0-9]+(.[0-9]+)?\=[0-9]+(.[0-9]+)?>>")

choice_match = re.compile(r"([0-9]+ / [0-9]+)")


@lru_cache()
def clean_text(answer_txt):
    if "<div" in answer_txt:
        soup = BeautifulSoup(answer_txt)
        if soup.find("div", {"class": "markdown"}):
            clean_text = markdownify(answer_txt)
        elif soup.find("code"):
            clean_text = markdownify(answer_txt)
        else:
            clean_text = ""
            for para in soup.find_all("p"):
                clean_text = para.text + "\n"
            clean_text = clean_text.strip()

        if "Copy code`" in clean_text:
            clean_text = clean_text.replace("Copy code`", "")
    else:
        clean_text = answer_txt
    # remove gmsk invalid format due to markdown
    clean_text = clean_prefix(clean_text)
    return invalid_code.sub("", clean_text)


def clean_prefix(clean_text):
    if choice_match.search(clean_text):  # remove choice
        res = choice_match.search(clean_text)
        if res.start() == 0:
            clean_text = clean_text[res.end() :]
    return clean_text


def load_chatdataset(tokenizer, path="dataset/sharegpt", input_limit=400):
    pairs = []
    dedupe = []
    for json_file in tqdm(glob.glob(os.path.join(path, "*.json"))):
        with open(json_file, "r") as f:
            result = json.load(f)
        items = result["pageProps"]["content"]["items"]
        if isinstance(items[0], str):
            # weird incorrect format
            continue

        first_txt = items[0]["value"] + items[1]["value"]
        if len(items) > 2:
            first_txt += items[2]["value"]
        if first_txt in dedupe:
            continue
        dedupe.append(first_txt)

        # start process text
        for idx in range(0, len(items), 2):
            question = items[idx]
            if isinstance(question, str):
                # weird incorrect format
                continue

            if len(question["value"]) < input_limit and idx == 0:
                question_txt = question["value"]
            elif idx > 2 and len(question["value"]) < input_limit:
                convo = items[idx - 2 : idx + 1]
                text = " ".join(
                    [
                        " <bot> " + clean_text(c["value"])
                        if c["from"] == "gpt"
                        else " <sep> " + c["value"].strip()
                        for c in convo
                    ]
                )
                start_idx = idx - 2

                while len(text) < 500 and start_idx >= 0:
                    convo = items[max(start_idx, 0) : idx + 1]
                    text = " ".join(
                        [
                            " <bot> " + clean_text(c["value"])
                            if c["from"] == "gpt"
                            else " <sep> " + c["value"].strip()
                            for c in convo
                        ]
                    )
                    start_idx -= 1

                convo = items[max(start_idx, 0) : idx + 1]
                text = " ".join(
                    [
                        " <bot> " + clean_text(c["value"])
                        if c["from"] == "gpt"
                        else " <sep> " + c["value"].strip()
                        for c in convo
                    ]
                )
                if len(text) <= 500:
                    question_txt = text
                else:
                    question_txt = tokenizer.decode(
                        tokenizer(question["value"])["input_ids"][:input_limit]
                    ).replace("</s>", "")
            else:
                question_txt = tokenizer.decode(
                    tokenizer(question["value"])["input_ids"][:input_limit]
                ).replace("</s>", "")

            if choice_match.search(question_txt):  # remove choice
                res = choice_match.search(question_txt)
                if res.start() == 0:
                    question_txt = question_txt[res.end() :]
            if "!Contents may violate our content policy" in question_txt:
                continue
            if (idx + 1) >= len(items):
                continue

            response = items[idx + 1]
            if response["from"] != "gpt":
                if (idx + 2) < len(items):
                    question_txt = response["value"]
                    answer_txt = items[idx + 2]["value"]
                else:
                    print(response)
                    continue
            else:
                assert response["from"] == "gpt"
                answer_txt = clean_text(response["value"])

            if len(question_txt) and len(answer_txt):
                question_txt = clean_prefix(question_txt)
                # prevent newline from eaten by mt
                pairs.append(
                    (
                        question_txt.replace("\n", "\\n") + " <bot>",
                        answer_txt.replace("\n", "\\n"),
                    )
                )
            question_txt = ""
            answer_txt = ""
    del dedupe

    return pairs


class ChatGPT(Dataset):
    def __init__(
        self,
        datapath="dataset/sharegpt",
        input_max_length=650,
        max_length=650,
        tokenizer_name="google/mt5-small",
    ) -> None:
        super().__init__()
        if isinstance(tokenizer_name, str):
            self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        else:
            self.tokenizer = tokenizer_name
        self.pairs = load_chatdataset(self.tokenizer, datapath)
        self.input_max_length = input_max_length
        self.max_length = max_length

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        prompt, answer = self.pairs[idx]
        input_encodings = self.tokenizer(
            prompt, max_length=self.input_max_length, truncation=True
        )

        with self.tokenizer.as_target_tokenizer():
            target_encodings = self.tokenizer(
                answer, max_length=self.max_length, truncation=True
            )

        return {
            "input_ids": input_encodings["input_ids"],
            "attention_mask": input_encodings["attention_mask"],
            "labels": target_encodings["input_ids"],
        }


class MLQA(Dataset):

    language_question_text = {
        "ar": "سؤال",
        "de": "Frage",
        "vi": "câu hỏi",
        "zh": "问题",
        "en": "Question",
        "es": "Pregunta",
        "hi": "प्रश्न",
    }

    def __init__(
        self, tokenizer_name="google/mt5-small", input_max_length=400, max_length=128
    ) -> None:
        super().__init__()
        if isinstance(tokenizer_name, str):
            self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        else:
            self.tokenizer = tokenizer_name
        self.input_max_length = input_max_length
        self.max_length = max_length
        dataset_splits = [
            "mlqa.de.de",
            "mlqa.de.vi",
            "mlqa.de.zh",
            "mlqa.de.en",
            "mlqa.hi.ar",
            "mlqa.de.es",
            "mlqa.de.hi",
            "mlqa.vi.ar",
            "mlqa.vi.de",
            "mlqa.vi.vi",
            "mlqa.vi.en",
            "mlqa.vi.es",
            "mlqa.vi.hi",
            "mlqa.zh.ar",
            "mlqa.zh.de",
            "mlqa.zh.zh",
            "mlqa.zh.en",
            "mlqa.zh.es",
            "mlqa.zh.hi",
            "mlqa.en.en",
            "mlqa.en.ar",
            "mlqa.en.de",
            "mlqa.en.vi",
            "mlqa.en.zh",
            "mlqa.hi.de",
            "mlqa.es.vi",
            "mlqa.es.zh",
            "mlqa.es.en",
            "mlqa.es.es",
            "mlqa.es.hi",
            "mlqa.hi.en",
            "mlqa.hi.es",
            "mlqa.hi.hi",
            "mlqa.en.hi",
            "mlqa.es.ar",
            "mlqa.hi.zh",
            "mlqa.ar.es",
            "mlqa.es.de",
            "mlqa.hi.vi",
            "mlqa.en.es",
            "mlqa.zh.vi",
            "mlqa.vi.zh",
        ]
        self.pairs = []
        for split in dataset_splits:
            datasets = load_dataset("mlqa", split)
            question_lang = split.split(".")[-1]
            mid_token = " {}:".format(self.language_question_text[question_lang])
            for key, dataset in datasets.items():
                for row in dataset:
                    prompts = row["context"] + mid_token + row["question"]
                    answer = " ".join(row["answers"]["text"])
                    self.pairs.append((prompts, answer))

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, index):
        prompt, answer = self.pairs[index]
        input_encodings = self.tokenizer(
            prompt, max_length=self.input_max_length, truncation=True
        )

        with self.tokenizer.as_target_tokenizer():
            target_encodings = self.tokenizer(
                answer, max_length=self.max_length, truncation=True
            )

        return {
            "input_ids": input_encodings["input_ids"],
            "attention_mask": input_encodings["attention_mask"],
            "labels": target_encodings["input_ids"],
        }


class Text2Code(Dataset):
    def __init__(
        self, tokenizer_name="google/mt5-small", input_max_length=300, max_length=1024
    ) -> None:
        super().__init__()
        if isinstance(tokenizer_name, str):
            self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        else:
            self.tokenizer = tokenizer_name
        dataset_splits = [
            "Python-snippet-level",
            "Python-program-level",
            "C-snippet-level",
            "C-program-level",
            "Java-snippet-level",
            "Java-program-level",
            "Javascript-snippet-level",
            "Javascript-program-level",
            "Csharp-snippet-level",
            "Csharp-program-level",
            "C++-snippet-level",
            "C++-program-level",
            "PHP-snippet-level",
            "PHP-program-level",
        ]
        self.pairs = []
        self.max_length = max_length
        self.input_max_length = input_max_length
        for split in dataset_splits:
            dataset = load_dataset("codeparrot/xlcost-text-to-code", split)["train"]
            for row in dataset:
                prompts = row["text"] + " " + "write the code in " + split.split("-")[0]
                answer = "```\n" + " ".join(row["code"]) + "\n```"
                self.pairs.append((prompts, answer))

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, index):
        prompt, answer = self.pairs[index]
        input_encodings = self.tokenizer(
            prompt, max_length=self.input_max_length, truncation=True
        )

        with self.tokenizer.as_target_tokenizer():
            target_encodings = self.tokenizer(
                answer, max_length=self.max_length, truncation=True
            )

        return {
            "input_ids": input_encodings["input_ids"],
            "attention_mask": input_encodings["attention_mask"],
            "labels": target_encodings["input_ids"],
        }


class UNnatural(Dataset):
    def __init__(
        self, tokenizer_name="google/mt5-small", input_max_length=650, max_length=650
    ) -> None:
        super().__init__()
        if isinstance(tokenizer_name, str):
            self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        else:
            self.tokenizer = tokenizer_name
        self.pairs = []
        self.max_length = max_length
        self.input_max_length = input_max_length
        dataset = load_dataset("mrm8488/unnatural-instructions-core")["train"]
        for row in dataset:
            for data in row["instances"]:
                prompts = data["instruction_with_input"]
                answer = data["output"]
                self.pairs.append((prompts, answer))

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, index):
        prompt, answer = self.pairs[index]
        input_encodings = self.tokenizer(
            prompt, max_length=self.input_max_length, truncation=True
        )

        with self.tokenizer.as_target_tokenizer():
            target_encodings = self.tokenizer(
                answer, max_length=self.max_length, truncation=True
            )

        return {
            "input_ids": input_encodings["input_ids"],
            "attention_mask": input_encodings["attention_mask"],
            "labels": target_encodings["input_ids"],
        }


if __name__ == "__main__":
    import shutil

    from torch.utils.data import DataLoader
    from transformers import AutoModelForSeq2SeqLM

    for json_file in glob.glob("dataset/sharegpt/*.json"):
        json_basename = os.path.basename(json_file)
        if not os.path.exists(
            "dataset/sharegpt_train/" + json_basename
        ) and not os.path.exists("dataset/sharegpt_val/" + json_basename):
            shutil.copyfile(json_file, "dataset/sharegpt_train/" + json_basename)

    # tokenizer = AutoTokenizer.from_pretrained("google/mt5-small")

    model = AutoModelForSeq2SeqLM.from_pretrained("google/mt5-small")
    # dataset = ChatGPT('dataset/sharegpt_val')
    # dataset = ChatGPT('dataset/sharegpt_train')
    # print(len(dataset))
    dataset = UNnatural()
    print(len(dataset))
    dataloader = DataLoader(
        dataset,
        collate_fn=DataCollatorForSeq2Seq(dataset.tokenizer, model, max_length=1024),
        batch_size=128,
        num_workers=10,
    )
    for batch in dataloader:
        # batch = batch.to('cuda')
        # print(batch.__dict__)
        # print(batch['attention_mask'])
        # print(batch.keys())
        print(batch["labels"].shape)
        # break
        # print(batch['decoder_input_ids'][0])
