import torch
import re
from tqdm import tqdm
from transformers import (
    AutoModelForSequenceClassification, AutoModel, AutoTokenizer, AutoModelForSeq2SeqLM
)
from datasets import load_dataset
re_reference_remove = re.compile(r'\[([0-9])+\]')

# ideally we should generate results from multiple model
pretrain_model = 'google/flan-t5-base'
use_cuda = True
num_top_k = 5
model = AutoModelForSeq2SeqLM.from_pretrained(pretrain_model).eval()
tokenizer = AutoTokenizer.from_pretrained(pretrain_model)
dataset = load_dataset("openai/webgpt_comparisons")
critic = AutoModelForSequenceClassification.from_pretrained("google/electra-large_critic/checkpoint_e3").eval()
critic_tokenizer = AutoTokenizer.from_pretrained("google/electra-large_critic/checkpoint_e3")
if __name__ == "__main__":
    import json
    if use_cuda:
        critic = critic.half().cuda()
        model = model.half().cuda()

    with open('generated_noisy_text_v2.jsonl', 'a') as fout, torch.no_grad():
        for idx, row in tqdm(enumerate(dataset['train']), total=len(dataset['train'])):
            question = row['question']['full_text']
            inputs = tokenizer(question, return_tensors="pt")
            if use_cuda:
                inputs = {key: value.cuda() for key, value in inputs.items() }
            outputs = model.generate(**inputs, max_length=128,
                do_sample=True,
                top_k=50,
                num_return_sequences=10,
                early_stopping=True
            )
            texts = tokenizer.batch_decode(outputs, skip_special_tokens=True)
            answers = texts #[row['answer_0'], row['answer_1']]+texts
            inputs = critic_tokenizer([question]*len(answers), answers, return_tensors="pt", padding=True, truncation=True)
            if use_cuda:
                inputs = {key: value.cuda() for key, value in inputs.items() }
            ranks = critic(**inputs).logits.flatten()
            top_k = torch.topk(ranks, num_top_k, 0).indices
            fout.write(json.dumps({'idx': idx, 'question': question, 'negatives': [answers[idx] for idx in top_k] })+'\n')

