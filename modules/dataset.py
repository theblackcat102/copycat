import os
import glob
from tqdm import tqdm
import ujson as json
from markdownify import markdownify
from functools import lru_cache
from bs4 import BeautifulSoup
from torch.utils.data import Dataset
from transformers import AutoTokenizer, DataCollatorForSeq2Seq


@lru_cache()
def clean_text(answer_txt):
    if '<div' in answer_txt:
        soup = BeautifulSoup(answer_txt)
        if soup.find('div', {'class': 'markdown'}):
            clean_text = markdownify(answer_txt)
        elif soup.find('code'):
            clean_text = markdownify(answer_txt)
        else:
            clean_text = ''
            for para in soup.find_all('p'):
                clean_text = para.text+'\n'
            clean_text = clean_text.strip()
    else:
        clean_text = answer_txt
    return clean_text

def load_dataset(tokenizer, path='dataset/sharegpt', input_limit=400):
    pairs = []
    dedupe = []
    for json_file in tqdm(glob.glob(os.path.join(path, '*.json'))):
        with open(json_file, 'r') as f:
            result = json.load(f)
        items = result['pageProps']['content']['items']
        if isinstance(items[0], str):
            # weird incorrect format
            continue

        first_txt = items[0]['value']+items[1]['value']
        if len(items) > 2:
            first_txt += items[2]['value']
        if first_txt in dedupe:
            continue
        dedupe.append(first_txt)

        # start process text
        for idx in range(0, len(items), 2):
            question = items[idx]
            if isinstance(question, str):
                # weird incorrect format
                continue

            if len(question['value']) < input_limit and idx == 0:
                question_txt = question['value']
            elif idx > 2 and len(question['value']) < input_limit:
                convo = items[idx-2:idx+1]
                text = ' <sep> '.join([ clean_text(c['value']) if c['from'] == 'gpt' else c['value'].strip() for c in convo ])
                start_idx = idx-2

                while len(text) < 500 and start_idx >= 0:
                    convo = items[max(start_idx, 0):idx+1]
                    text = '<sep>'.join([ clean_text(c['value']) if c['from'] == 'gpt' else c['value'].strip() for c in convo ])
                    start_idx -= 1

                convo = items[max(start_idx, 0):idx+1]
                text = '<sep>'.join([ clean_text(c['value']) if c['from'] == 'gpt' else c['value'].strip() for c in convo ])
                if len(text) <= 500:
                    question_txt = text
                else:
                    question_txt = tokenizer.decode(tokenizer(question['value'])['input_ids'][:input_limit]).replace('</s>','')
            else:
                question_txt = tokenizer.decode(tokenizer(question['value'])['input_ids'][:input_limit]).replace('</s>','')

            response = items[idx+1]
            assert response['from'] == 'gpt'
            answer_txt = clean_text(response['value'])
            if len(question_txt) and len(answer_txt):
                pairs.append((question_txt, answer_txt))

    del dedupe

    return pairs

class ChatGPT(Dataset):

    def __init__(self, datapath='dataset/sharegpt', tokenizer_name="google/mt5-small") -> None:
        super().__init__()
        if isinstance(tokenizer_name, str):
            self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        else:
            self.tokenizer = tokenizer_name
        self.pairs = load_dataset(self.tokenizer, datapath)
    
    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        prompt, answer = self.pairs[idx]
        input_encodings = self.tokenizer(prompt, max_length=301,
                                        truncation=True)

        with self.tokenizer.as_target_tokenizer():
            target_encodings = self.tokenizer(answer, max_length=401,
                                        truncation=True)

        return {
                "input_ids": input_encodings["input_ids"],
                "attention_mask": input_encodings["attention_mask"],
                "labels": target_encodings["input_ids"]
                }

if __name__ == "__main__":
    import shutil
    from torch.utils.data import DataLoader
    from transformers import AutoModelForSeq2SeqLM

    for json_file in glob.glob('dataset/sharegpt/*.json'):
        json_basename = os.path.basename(json_file)
        if not os.path.exists('dataset/sharegpt_train/'+json_basename) and not \
            os.path.exists('dataset/sharegpt_val/'+json_basename):
            shutil.copyfile(json_file, 'dataset/sharegpt_train/'+json_basename)

    tokenizer = AutoTokenizer.from_pretrained("google/mt5-small")

    model = AutoModelForSeq2SeqLM.from_pretrained("google/mt5-small")
    dataset = ChatGPT('dataset/sharegpt_val')
    dataset = ChatGPT('dataset/sharegpt_train')
    print(len(dataset))
    dataloader = DataLoader(dataset, collate_fn=DataCollatorForSeq2Seq(dataset.tokenizer, model,max_length=402), batch_size=128, num_workers=10)
    for batch in dataloader:
        batch = batch.to('cuda')
        print(batch.keys())
        break
    #     print(batch['input_ids'][0])
    #     print(batch['decoder_input_ids'][0])
