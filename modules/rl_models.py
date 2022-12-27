import copy
from pathlib import Path
from collections import namedtuple
import torch
from torch import nn
from tqdm import tqdm
from beartype import beartype
from typing import Tuple, Optional
from modules.utils import eval_decorator



PPOActionCriticReturn = namedtuple('PPOActionCriticReturn', [
    'actions',
    'sequence',
    'mask',
    'prompt_mask',
    'action_logits',
    'values'
])

@beartype
class ActorCritic(nn.Module):
    def __init__(
        self,
        model: nn.Module,
        critic: nn.Module,
    ):
        super().__init__()
        self.actor_palm = model

        self.critic = critic


    def actor_parameters(self):
        if not self.actor_lora:
            return self.actor_palm.parameters()

        return [
            *self.actor_palm.finetune_parameters(self.actor_lora_scope)
        ]

    def critic_parameters(self):
        return self.critic.parameters()

    @torch.no_grad()
    @eval_decorator
    def generate(
        self,
        state, # prompt
        max_seq_len,
        eos_token = None,
        return_values = False,
        **kwargs
    ):
        actions = self.actor_palm.generate(
            max_seq_len,
            prompt = state,       
            eos_token = eos_token,     
            finetune_scope = self.actor_lora_scope,
            use_tqdm = True,
            **kwargs
        )

        sequence = torch.cat((state, actions), dim = -1)
        action_len = actions.shape[-1]
        state_len = state.shape[-1]

        prompt_mask = torch.arange(sequence.shape[-1], device = state.device) < state_len
        prompt_mask = repeat(prompt_mask, 'n -> b n', b = sequence.shape[0])

        mask = None
        if exists(eos_token):
            mask = ((sequence == eos_token).cumsum(dim = -1) == 0)
            mask = F.pad(mask, (1, -1), value = True) # include eos token
        # we pair the question and answer into 
        action_logits, value = self.forward(
            sequence,
            mask = mask,
            return_values = return_values
        )
        print(action_logits.shape, value.shape) 

        return PPOActionCriticReturn(
            actions,
            sequence,
            mask,
            prompt_mask,
            action_logits,
            value
        )

    def forward(
        self,
        x,
        mask = None,
        return_values = True
    ):  
        # we use this action_logits for finetuning later?
        action_logits = self.actor_palm(
            x,
        )

        if not return_values:
            return action_logits, None

        # predict the reward value
        values = self.critic()
        return action_logits, values
