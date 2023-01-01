# -*- coding: utf-8 -*-
from colossalai.amp import AMP_TYPE

BATCH_SIZE = 2
NUM_EPOCHS = 50
LR = 2e-5
CONFIG = dict(fp16=dict(mode=AMP_TYPE.TORCH))


fp16 = dict(mode=AMP_TYPE.NAIVE)
