from typing import Optional

import torch
from colossalai.nn.optimizer.cpu_adam import CPUAdam
from tqdm import tqdm

from tensornvme import DiskOffloader
from benchmark_adam import gpt2_xl

N_WARMUP = 2
N_ACTIVATE = 4


class NVMECPUAdam(CPUAdam):
    def __init__(self, model_params,
                 lr=1e-3,
                 bias_correction=True,
                 betas=(0.9, 0.999),
                 offloader: Optional[DiskOffloader] = None,
                 prefetch: int = 0,
                 vecio: bool = False,
                 eps=1e-8,
                 weight_decay=0,
                 adamw_mode=True,
                 simd_log=False):
        super(NVMECPUAdam, self).__init__(
            model_params, lr, bias_correction, betas, eps, weight_decay, adamw_mode, simd_log)

        self.offloader = offloader
        self.prefetch = prefetch
        self.vecio = vecio
        # init states
        for group in self.param_groups:
            for p in group['params']:
                if p.requires_grad:
                    state = self.state[p]
                    state['step'] = 0
                    state['exp_avg'] = torch.zeros_like(p)
                    state['exp_avg_sq'] = torch.zeros_like(p)
                    if self.offloader is None:
                        continue
                    if vecio:
                        self.offloader.sync_writev(
                            [state['exp_avg'], state['exp_avg_sq']])
                    else:
                        self.offloader.sync_write(state['exp_avg'])
                        self.offloader.sync_write(state['exp_avg_sq'])

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for _, group in enumerate(self.param_groups):
            self._init_step(group['params'])
            for p_i, p in enumerate(group['params']):

                state = self.state[p]
                target_device = p.device
                state['step'] += 1
                beta1, beta2 = group['betas']

                if target_device.type == 'cpu':
                    assert p.data.numel() == p.grad.data.numel(
                    ), "parameter and gradient should have the same size"
                    assert state['exp_avg'].device.type == 'cpu', "exp_avg should stay on cpu"
                    assert state['exp_avg_sq'].device.type == 'cpu', "exp_avg should stay on cpu"
                    self._pre_step(p_i, group['params'])
                    self.cpu_adam_op.adam_update(self.opt_id, state['step'], group['lr'], beta1, beta2, group['eps'],
                                                 group['weight_decay'], group['bias_correction'], p.data, p.grad.data,
                                                 state['exp_avg'], state['exp_avg_sq'], -1)
                    self._post_step(p_i, group['params'])
                elif target_device.type == 'cuda':
                    assert state['exp_avg'].device.type == 'cuda', "exp_avg should stay on cuda"
                    assert state['exp_avg_sq'].device.type == 'cuda', "exp_avg should stay on cuda"

                    bias_correction1 = 1 - beta1 ** state['step']
                    bias_correction2 = 1 - beta2 ** state['step']

                    # adam on cuda
                    self.torch_adam_update(p.data, p.grad.data, state['exp_avg'], state['exp_avg_sq'], group['lr'],
                                           beta1, beta2, group['eps'], group['weight_decay'], bias_correction1,
                                           bias_correction2, self.adamw_mode)
                else:
                    raise RuntimeError
        return loss

    def _init_step(self, params):
        if self.offloader is not None and self.prefetch > 0:
            for p in params[:self.prefetch]:
                state = self.state[p]
                if self.vecio:
                    self.offloader.sync_readv(
                        [state['exp_avg'], state['exp_avg_sq']])
                else:
                    self.offloader.sync_read(state['exp_avg'])
                    self.offloader.sync_read(state['exp_avg_sq'])

    def _pre_step(self, idx, params):
        if self.offloader is None:
            return
        if self.prefetch > 0:
            if idx % self.prefetch == 0:
                self.offloader.sync_read_events()
                if idx + self.prefetch < len(params):
                    for prefetch_p in params[idx + self.prefetch:idx + self.prefetch * 2]:
                        prefetch_state = self.state[prefetch_p]
                        if self.vecio:
                            self.offloader.async_readv(
                                [prefetch_state['exp_avg'], prefetch_state['exp_avg_sq']])
                        else:
                            self.offloader.async_read(
                                prefetch_state['exp_avg'])
                            self.offloader.async_read(
                                prefetch_state['exp_avg_sq'])
        else:
            state = self.state[params[idx]]
            if self.vecio:
                self.offloader.sync_readv(
                    [state['exp_avg'], state['exp_avg_sq']])
            else:
                self.offloader.sync_read(state['exp_avg'])
                self.offloader.sync_read(state['exp_avg_sq'])

    def _post_step(self, idx, params):
        if self.offloader is None:
            return
        state = self.state[params[idx]]
        if self.prefetch > 0:
            if idx % self.prefetch == 0:
                self.offloader.sync_write_events()
            if self.vecio:
                self.offloader.async_writev(
                    [state['exp_avg'], state['exp_avg_sq']])
            else:
                self.offloader.async_write(state['exp_avg'])
                self.offloader.async_write(state['exp_avg_sq'])
        else:
            if self.vecio:
                self.offloader.sync_writev(
                    [state['exp_avg'], state['exp_avg_sq']])
            else:
                self.offloader.sync_write(state['exp_avg'])
                self.offloader.sync_write(state['exp_avg_sq'])


def run_adam(model: torch.nn.Module, nvme_offload: bool, backend: str, prefetch: int, vecio: bool):
    offloader = None
    if nvme_offload:
        offloader = DiskOffloader('.', 8, backend=backend)
    params = list(model.cpu().parameters())
    for _, p in enumerate(params):
        if p.grad is None and p.requires_grad:
            p.grad = torch.rand_like(p.data, dtype=torch.float)
    optimizer = NVMECPUAdam(
        params, 1e-3, offloader=offloader, prefetch=prefetch, vecio=vecio)
    for p in model.parameters():
        p.grad = torch.rand_like(p)
    for _ in range(N_WARMUP):
        optimizer.step()
    if not nvme_offload:
        desc = 'CPU'
        postfix = None
    else:
        desc = 'NVME'
        postfix = {'backend': backend, 'prefetch': prefetch, 'vecio': vecio}
    for _ in tqdm(range(N_ACTIVATE), desc=desc, postfix=postfix):
        optimizer.step()


if __name__ == '__main__':
    model = gpt2_xl()
    with torch.no_grad():
        run_adam(model, False, 'uring', 0, False)
        run_adam(model, True, 'uring', 0, False)
        run_adam(model, True, 'uring', 0, True)
        run_adam(model, True, 'uring', 1, False)
        run_adam(model, True, 'uring', 1, True)
        run_adam(model, True, 'uring', 2, False)
        run_adam(model, True, 'uring', 2, True)
        run_adam(model, True, 'uring', 4, False)
        run_adam(model, True, 'uring', 4, True)