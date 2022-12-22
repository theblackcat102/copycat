import os
import glob
import json
import torch
import numpy as np
import pytorch_lightning as pl
from torch import nn
from torch.optim import AdamW
import torch.nn.functional as F
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import Dataset
from datasets import load_dataset
from pytorch_lightning.trainer.supporters import CombinedLoader
from pytorch_lightning.plugins import DDPPlugin
from pytorch_lightning import loggers as pl_loggers
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor

class WebGPTReward(Dataset):
    def __init__(self, mode='train', index_cache='dataset/webgpt_train_idx.pt') -> None:
        super().__init__()
        dataset = load_dataset("openai/webgpt_comparisons")
        if os.path.exists(index_cache):
            train_idx = torch.load(index_cache)
        else:
            train_idx = np.random.choice(range(len(dataset['train'])), int(len(dataset['train'])*0.8), replace=False)
            torch.save(set(train_idx.tolist()), index_cache)
        self.dataset = []
        for idx, row in enumerate(dataset['train']):
            if mode == 'train' and idx in train_idx:
                self.dataset.append({'question': row['question']['full_text'], 'pos': row['answer_0'], 'neg':row['answer_1'] })
            elif idx not in train_idx and mode != 'train':
                self.dataset.append({'question': row['question']['full_text'], 'pos': row['answer_0'], 'neg':row['answer_1'] })

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        row = self.dataset[index]
        return row['question'], row['pos'], row['neg']

class CollateFN():
    def __init__(self, pretrain_name, max_length=400) -> None:
        self.tokenizer = AutoTokenizer.from_pretrained(pretrain_name)
        self.max_length = max_length

    def __call__(self, batch):
        questions = []
        pos_sentences = []
        neg_sentences = []
        for (question, pos, neg) in batch:
            questions.append(question)
            pos_sentences.append(pos)
            neg_sentences.append(neg)

        return self.tokenizer(questions, pos_sentences, return_tensors='pt', max_length=self.max_length, padding=True, truncation=True),\
            self.tokenizer(questions, neg_sentences, return_tensors='pt', max_length=self.max_length, padding=True, truncation=True)


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
        self.lr = lr
        self.log_sigmoid = nn.LogSigmoid()
        self.num_training_steps = total_iterations
        self.num_warmup_steps = num_warmup_steps

    def forward(self, batch):
        return self.model(**batch)

    def training_step(self, batch, batch_idx):
        pos_batch, neg_batch = batch
        pos_output = self.forward(pos_batch)
        neg_output = self.forward(neg_batch)
        # print(pos_output.logits, pos_output.logits[:10])
        loss = -self.log_sigmoid(pos_output.logits - neg_output.logits)
        loss = loss.mean()
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
        self.log('val/accuracy', correct/total)

    def configure_optimizers(self):
        optimizer = AdamW(self.parameters(), lr=self.lr)
        scheduler = get_linear_schedule_with_warmup(optimizer, self.num_warmup_steps, self.num_training_steps)
        return [optimizer], [{ 'scheduler': scheduler, 'interval': 'step', 'name': 'warmup_decay' }]


if __name__ == "__main__":
    pretrain_name='bigscience/bloomz-560m'
    model_name = pretrain_name.split('/')[-1]
    epochs = 50
    lr = 1e-6
    
    val_dataset = WebGPTReward(mode='val')
    print(len(val_dataset))
    train_dataset = WebGPTReward(mode='train')
    print(len(train_dataset))
    val_dataloader = DataLoader(val_dataset, collate_fn=CollateFN(pretrain_name, max_length=350), batch_size=8)
    train_dataloader = DataLoader(train_dataset,
        collate_fn=CollateFN(pretrain_name, max_length=220),
        batch_size=16, shuffle=True
    )

    total_iterations = len(train_dataloader)*epochs
    model = RewardModel(pretrain_name,
            num_warmup_steps=1000,
            total_iterations=total_iterations,
            lr=lr
        )

    lr_monitor = LearningRateMonitor(logging_interval='step')
    checkpoint_callback = ModelCheckpoint(monitor="val_loss",
        dirpath="checkpoints/"+model_name,
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