'''

    Differences from WebGPT method:

    1. Use crossentropy yield much better results. WebGPT uses rank loss, not sure why (Sec 3.5)

    2. Overfitting is pretty easy

    C.2 "176B RMs have an accuracy of 69.6 ± 0.9% on predicting the preferences of labelers in the held-out group"
    
    Bloomz 556M

        Alot of parameters was spent in embedding for other languages, this dataset is pure en, so ignore for now

    Roberta

        roberta-large requires a larger learning rate : 6e-6 and epochs of 4-6, batch size of 16

    GPT2:
        gpt2-large seems to be quite unstable (due to limited VRAM)

        gpt2-base : truncate half the layers of gpt2-large seems to be much stable due to larger batch size

        GPT2 series conclusion : model is too unstable, even validation loss isn't converging


    TLDR;




'''
import os
import torch
import re
import json
import random
import math
import numpy as np
import pytorch_lightning as pl
from torch import nn
from datasets import load_dataset
from torch.optim import AdamW
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import Dataset
from pytorch_lightning import loggers as pl_loggers
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor

re_reference_remove = re.compile(r'\[([0-9])+\]')
def return_format(row):
    if row['score_0'] >= row['score_1']:
        # remove this to prevent information leak, since we are not using reference
        return {
                'question': row['question']['full_text'],
                     'pos': re_reference_remove.sub('', row['answer_0']),
                     'neg': re_reference_remove.sub('', row['answer_1'])
                }

    return {
            'question': row['question']['full_text'],
                 'pos': re_reference_remove.sub('', row['answer_1']),
                 'neg': re_reference_remove.sub('', row['answer_0'])
            }


class WebGPTReward(Dataset):
    def __init__(self, mode='train', index_cache='dataset/webgpt_train_idx.pt', additional_dataset=None) -> None:
        super().__init__()
        dataset = load_dataset("openai/webgpt_comparisons")
        if os.path.exists(index_cache):
            train_idx = torch.load(index_cache)
        else:
            train_idx = np.random.choice(range(len(dataset['train'])), int(len(dataset['train'])*0.8), replace=False)
            torch.save(set(train_idx.tolist()), index_cache)
        self.dataset = []
        self.dataset_index = []
        for idx, row in enumerate(dataset['train']):
            if mode == 'train' and idx in train_idx:
                self.dataset.append(return_format(row))
                self.dataset_index.append(idx)
            elif idx not in train_idx and mode != 'train':
                self.dataset.append(return_format(row))

        # since this dataset was generated from 176B GPT-3
        # we needed some more sample generated from other base model
        # such as mT0 or Flan-T5
        self.sample_additional = False
        if additional_dataset is not None:
            self.sample_additional = mode == 'train'
            self.additional = {}
            with open(additional_dataset, 'r') as f:
                for line in f:
                    row = json.loads(line)
                    if row['idx'] in train_idx:
                        self.additional[row['idx']] = row['negatives']
            for match_idx in train_idx:
                if match_idx in self.additional:
                    continue

                idx = match_idx-900
                while idx not in self.additional:
                    idx -= 1
                self.additional[match_idx] = self.additional[idx]


    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        row = self.dataset[index]
        if not self.sample_additional:
            return row['question'], row['pos'], row['neg']

        gen_neg = random.choice(self.additional[self.dataset_index[index]])
        return row['question'], row['pos'], row['neg'], gen_neg

class CollateFN():
    def __init__(self, pretrain_name, max_length=400) -> None:
        self.tokenizer = AutoTokenizer.from_pretrained(pretrain_name)
        if 'gpt2-' in pretrain_name:
            self.tokenizer.add_special_tokens({'pad_token': self.tokenizer.eos_token})
        self.max_length = max_length

    def __call__(self, batch):
        questions = []
        pos_sentences = []
        neg_sentences = []
        additional_neg = []
        for triplets in batch:
            questions.append(triplets[0])
            pos_sentences.append(triplets[1])
            neg_sentences.append(triplets[2])
            if len(triplets) > 3:
                additional_neg.append(triplets[3])

        batch = [self.tokenizer(questions, pos_sentences, return_tensors='pt', max_length=self.max_length, padding=True, truncation=True),\
                self.tokenizer(questions, neg_sentences, return_tensors='pt', max_length=self.max_length, padding=True, truncation=True)]
        if len(additional_neg) == 0: # evaluation
            return batch

        batch += [
            self.tokenizer(questions, additional_neg, return_tensors='pt', max_length=self.max_length, padding=True, truncation=True)
        ]
        return batch



def get_linear_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps, last_epoch=-1):
    def lr_lambda(current_step):
        learning_rate = max(0.0, 1. - (float(current_step) / float(num_training_steps)))
        learning_rate *= min(1.0, float(current_step) / float(num_warmup_steps))
        return learning_rate

    return LambdaLR(optimizer, lr_lambda, last_epoch)


class RewardModel(pl.LightningModule):

    def __init__(self, pretrain_name='bigscience/bloomz-560m',
        num_warmup_steps=1000,
        lr=3e-4,
        total_iterations=10000,
        ) -> None:
        super().__init__()
        self.model = AutoModelForSequenceClassification.from_pretrained(pretrain_name, num_labels=1, problem_type='regression')
        if 'gpt2-' in pretrain_name: # we will use eos as padding token
            self.model.config.pad_token_id = self.model.config.eos_token_id

        self.lr = lr
        # self.log_sigmoid = nn.LogSigmoid()
        self.xentropy = nn.CrossEntropyLoss()
        self.num_training_steps = total_iterations
        self.num_warmup_steps = num_warmup_steps

    def forward(self, batch):
        return self.model(**batch)

    def training_step(self, batch, batch_idx):
        pos_batch, neg_batch, neg_batch2 = batch
        pos_output = self.forward(pos_batch)
        neg_output = self.forward(neg_batch)
        neg_output2 = self.forward(neg_batch2)
        # print(pos_output.logits, pos_output.logits[:10])
        # loss = -self.log_sigmoid(pos_output.logits - neg_output.logits)
        # loss = loss.mean()
        loss = self.xentropy(torch.cat([pos_output.logits, neg_output.logits, neg_output2.logits], dim=-1),
                torch.zeros(pos_output.logits.shape[0], device=pos_output.logits.device, dtype=torch.long)
            )
        num_acc_examples = pos_batch.input_ids.shape[0]
        num_correct_examples = torch.count_nonzero(pos_output.logits > neg_output.logits).item()
        self.log('train/acc', num_correct_examples/num_acc_examples)
        self.log('train/loss', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        pos_batch, neg_batch = batch
        pos_output = self.forward(pos_batch)
        neg_output = self.forward(neg_batch)
        loss = -torch.log(torch.sigmoid(pos_output.logits - neg_output.logits))
        loss = loss.mean()
        num_acc_examples = pos_batch.input_ids.shape[0]
        num_correct_examples = torch.count_nonzero(pos_output.logits > neg_output.logits).item()
        return {
            'loss': loss.item(),
            'total': num_acc_examples,
            'correct': num_correct_examples
        }


    def validation_epoch_end(self, outs):
        total = np.sum([v['total'] for v in outs ])
        correct = np.sum([v['correct'] for v in outs ])
        avg_loss = np.sum([v['loss'] for v in outs ])/len(outs)
        self.log('val/loss', avg_loss)
        self.log('val_loss', avg_loss) # this is needed for checkpoint saving
        self.log('val/accuracy', correct/total)

    def configure_optimizers(self):
        optimizer = AdamW(self.parameters(), lr=self.lr)
        scheduler = get_linear_schedule_with_warmup(optimizer, self.num_warmup_steps, self.num_training_steps)
        return [optimizer], [{ 'scheduler': scheduler, 'interval': 'step', 'name': 'warmup_decay' }]


def convert_checkpoint2huggingface(checkpoint_path, pretrain_name, output_name):
    model = AutoModelForSequenceClassification.from_pretrained(pretrain_name, num_labels=1, problem_type='regression')
    model_state = model.state_dict()
    tokenizer = AutoTokenizer.from_pretrained(pretrain_name)
    tokenizer.save_pretrained(output_name)
    state_dict = torch.load(checkpoint_path, map_location='cpu')
    for key, tensor in state_dict.items():
        if key.startswith('model.'):
            new_key = key[len('model.'):]
            model_state[new_key] = tensor

    model.load_state_dict(model_state)
    model.save_pretrained(output_name)


if __name__ == "__main__":
    pretrain_name='bigscience/bloomz-560m'
    pretrain_name = 'roberta-large'
    # pretrain_name = 'gpt2-large'
    # pretrain_name = 'openai/gpt2-base'
    model_name = pretrain_name.split('/')[-1]
    epochs = 4 # webgpt RM finetuned for 2 epochs
    batch_size = 12
    # try to follow WebGPT RM batch size
    accumulate_grad_batches = math.ceil(64/batch_size)
    lr = 4e-6

    val_dataset = WebGPTReward(mode='val')
    print(len(val_dataset))
    train_dataset = WebGPTReward(mode='train', additional_dataset='generated_noisy_text.jsonl')
    print(len(train_dataset))
    val_dataloader = DataLoader(val_dataset, collate_fn=CollateFN(pretrain_name, max_length=350), batch_size=batch_size*2)
    train_dataloader = DataLoader(train_dataset,
        collate_fn=CollateFN(pretrain_name, max_length=220),
        batch_size=batch_size, shuffle=True, num_workers=5
    )

    total_iterations = len(train_dataloader)*epochs
    model = RewardModel(pretrain_name,
            num_warmup_steps=200,
            total_iterations=total_iterations,
            lr=lr
        )

    lr_monitor = LearningRateMonitor(logging_interval='step')
    checkpoint_callback = ModelCheckpoint(monitor="val_loss",
        dirpath="bigscience/"+model_name+'_critic',
        filename=model_name+"-{epoch:02d}-{val_loss:.2f}",
        save_top_k=3,
        mode="min"
    )

    logging_path = os.path.join('logging', model_name)
    version_number = 1
    version = 'version{:d}'.format(version_number)
    while os.path.exists(os.path.join(logging_path, version)):
        version_number +=1
        version = 'version{:d}'.format(version_number)

    trainer = pl.Trainer(
        max_epochs=epochs,
        accelerator='gpu',
        devices=1,
        gradient_clip_val=2,
        precision=16,
        accumulate_grad_batches=5,
        num_sanity_val_steps=2,
        strategy=None,
        callbacks=[checkpoint_callback, lr_monitor],
        logger=[
            pl_loggers.TensorBoardLogger(
                save_dir=os.getcwd(),
                version=version,
                name=logging_path
            ),
        ]
    )
    trainer.fit(model, train_dataloader, val_dataloader)