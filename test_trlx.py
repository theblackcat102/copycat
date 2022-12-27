'''

    Download full_data from unnatural instruction and unzip.

    https://github.com/orhonovich/unnatural-instructions/tree/main/data


    Some issues with fp16 with deepspeed - zero 3

    - T5-v1.1 doesn't work, numerical unstable

    - all t5 like model faces a similar issue of gradient scaling problem and never finish the training cycle

        - it is possible to do the same like my weight rescale method, but I would want to avoid this


'''
import os
import torch
import yaml
import random
import trlx
import json
from typing import List
from trlx.data.configs import TRLConfig
from tqdm import tqdm
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from datasets import load_dataset
from modules.utils import eval_decorator


default_config = yaml.safe_load(open("configs/ppo_config_t5.yml"))
class RM():

    def __init__(self, pretrain_model, max_length=300) -> None:
        self.model = AutoModelForSequenceClassification.from_pretrained(pretrain_model).half().cuda()
        self.model.eval()
        self.max_length = max_length
        self.tokenizer = AutoTokenizer.from_pretrained(pretrain_model)

    def __call__(self, samples):
        tokens = self.tokenizer(samples, return_tensors='pt', max_length=self.max_length, padding=True, truncation=True)
        tokens = { key: tensor.cuda() for key, tensor in tokens.items() }
        with torch.no_grad():
            return self.model(**tokens).logits.cpu().numpy().tolist()

    def reweight_model_logits(self, offset=None):
        if offset is None:
            import re

            re_reference_remove = re.compile(r'\[([0-9])+\]|\[([0-9])+,([0-9])+\]')
            dataset = load_dataset("openai/webgpt_comparisons")
            scores = []
            with torch.no_grad():
                for row in tqdm(dataset['train'], dynamic_ncols=True):
                    question = row['question']['full_text']
                    answer1 = re_reference_remove.sub('', row['answer_0'])
                    answer2 = re_reference_remove.sub('', row['answer_1'])
                    tokens = self.tokenizer([ question]*2, [answer1, answer2],
                        return_tensors='pt', max_length=self.max_length, padding=True, truncation=True)
                    tokens = { key: tensor.cuda() for key, tensor in tokens.items() }
                    scores.append(self.model(**tokens).logits)
            offset = torch.mean(torch.cat(scores))        
        state_dict = self.model.state_dict()
        state_dict['classifier.out_proj.bias'] -= offset
        self.model.load_state_dict(state_dict)
        print(offset)


def main(hparams={}):
    config = TRLConfig.update(default_config, hparams)

    if torch.cuda.is_available():
        device = int(os.environ.get("LOCAL_RANK", 0))
    else:
        device = -1

    rm_func = RM('theblackcat102/electra-large-webgpt-rm')
    rm_func.reweight_model_logits(0.2001)
    dataset = load_dataset("openai/webgpt_comparisons")
    prompts = [ row['question']['full_text'].replace('\n', '') for row in dataset['train']]
    with open('full_data.jsonl', 'r') as f:
        for line in f:
            data = json.loads(line)
            for instance in data['instances']:
                prompts.append(instance['instruction_with_input'])
    random.shuffle(prompts)

    trainer = trlx.train(reward_fn=rm_func, prompts=prompts, config=config)
    return trainer

if __name__ == "__main__":
    main()
