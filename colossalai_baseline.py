# -*- coding: utf-8 -*-
import math
import os
import time
from functools import partial

import colossalai
import torch
from colossalai.core import global_context as gpc
from colossalai.logging import disable_existing_loggers, get_dist_logger
from colossalai.nn.optimizer.gemini_optimizer import GeminiAdamOptimizer
from colossalai.tensor import ProcessGroup, ShardSpec
from colossalai.utils import get_dataloader
from colossalai.utils.model.colo_init_context import ColoInitContext
from colossalai.zero.init_ctx import ZeroInitContext
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

from modules.colossal_utils import (gemini_zero_dpp, get_mem_info,
                                    tensor_parallelize)
from modules.dataset import ChatGPT, DataCollatorForSeq2Seq
from modules.fixweights_utils import fix_rescale

pretrain_model = "google/mt5-large"
tokenizer = AutoTokenizer.from_pretrained(pretrain_model)
tokenizer.add_special_tokens({"sep_token": "<sep>"})


def parse_args():
    parser = colossalai.get_default_parser()
    parser.add_argument(
        "--tp_degree",
        type=int,
        default=1,
        help="Tensor Parallelism Degree. Valid when using colossalai as dist plan.",
    )
    parser.add_argument(
        "--placement",
        type=str,
        default="cpu",
        help="Placement Policy for Gemini. Valid when using colossalai as dist plan.",
    )
    parser.add_argument(
        "--shardinit",
        type=bool,
        default=False,
        help="Shard the tensors when init the model to shrink peak memory size on the assigned device. Valid when using colossalai as dist plan.",
    )
    args = parser.parse_args()
    return args


## Define the Model and Loss Based on Huggingface transformers GPT2LMHeadModel
class T5LMModel(nn.Module):
    def __init__(self, pretrain_name=pretrain_model, checkpoint=True):
        super().__init__()
        self.checkpoint = checkpoint
        model = AutoModelForSeq2SeqLM.from_pretrained(pretrain_name)
        model = fix_rescale(model, scale_down_factor=5)
        model.resize_token_embeddings(len(tokenizer))
        self.model = model
        if checkpoint:
            self.model.gradient_checkpointing_enable()

    def forward(self, input_ids, attention_mask, decoder_input_ids):
        # Only return lm_logits
        print(input_ids.shape, attention_mask.shape, decoder_input_ids.shape)
        return self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            decoder_input_ids=decoder_input_ids,
        )[0]


class LMLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.loss_fn = nn.CrossEntropyLoss()

    def forward(self, logits, labels):
        print(logits.shape, labels.shape)
        return self.loss_fn(logits.view(-1, logits.size(-1)), labels.view(-1))


def main():
    disable_existing_loggers()
    args = parse_args()

    colossalai.launch_from_torch(config="./config.py")
    logger = get_dist_logger()

    model = AutoModelForSeq2SeqLM.from_pretrained("google/mt5-large")
    tokenizer = AutoTokenizer.from_pretrained("google/mt5-large")
    tokenizer.add_special_tokens({"sep_token": "<sep>"})

    seq2seq_data_collator = DataCollatorForSeq2Seq(tokenizer, model, max_length=502)
    train_dataset = ChatGPT("dataset/sharegpt_train", tokenizer)
    train_dataloader = DataLoader(
        train_dataset,
        collate_fn=seq2seq_data_collator,
        batch_size=gpc.config.BATCH_SIZE,
        num_workers=10,
        pin_memory=True,
    )

    val_dataset = ChatGPT("dataset/sharegpt_val", tokenizer)
    test_dataloader = DataLoader(
        val_dataset,
        collate_fn=seq2seq_data_collator,
        batch_size=gpc.config.BATCH_SIZE,
        num_workers=10,
    )

    default_pg = ProcessGroup(tp_degree=args.tp_degree)
    default_dist_spec = ShardSpec([-1], [args.tp_degree]) if args.shardinit else None

    # build model for parallel
    with ColoInitContext(
        device="cpu", default_dist_spec=default_dist_spec, default_pg=default_pg
    ):
        model = T5LMModel()
    pg = default_pg
    # Tensor Parallelism (TP)
    # tensor_parallelize(model, pg)
    # Gemini + ZeRO DP, Note it must be used after TP
    model = gemini_zero_dpp(model, pg, args.placement)
    optimizer = GeminiAdamOptimizer(model, lr=gpc.config.LR, initial_scale=2**5)
    criterion = LMLoss()

    numel = sum([p.numel() for p in model.parameters()])
    logger.info(f"number of parameters {numel/1e6} M", ranks=[0])
    logger.info(get_mem_info(prefix="After init model, "), ranks=[0])

    # engine, train_dataloader, test_dataloader, _ = colossalai.initialize(
    #     model,
    #     optimizer,
    #     criterion,
    #     train_dataloader,
    #     test_dataloader,
    # )

    steps = 0
    NUM_STEPS = len(train_dataloader) * gpc.config.NUM_EPOCHS
    torch.cuda.synchronize()
    model.train()

    with tqdm(total=NUM_STEPS) as pbar:
        for e in range(gpc.config.NUM_EPOCHS):
            for batch in train_dataloader:
                device = torch.cuda.current_device()
                batch = batch.to(device)
                labels = batch.pop("labels").to()
                optimizer.zero_grad()
                outputs = model(
                    batch["input_ids"],
                    batch["attention_mask"],
                    batch["decoder_input_ids"],
                )
                loss = criterion(outputs.logits, labels)
                logger.info(
                    get_mem_info(prefix=f"[{steps}/{NUM_STEPS}] Forward "), ranks=[0]
                )
                optimizer.backward(loss)
                optimizer.step()
                pbar.update(1)

    torch.cuda.synchronize()


if __name__ == "__main__":
    main()
