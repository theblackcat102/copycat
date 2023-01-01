# -*- coding: utf-8 -*-
import colossalai
import psutil
import torch
from colossalai.nn.parallel import ZeroDDP
from colossalai.tensor import (ColoParameter, ComputePattern, ComputeSpec,
                               ProcessGroup, ReplicaSpec, ShardSpec)
from colossalai.utils import get_current_device
from packaging import version


def get_mem_info(prefix=""):
    return f"{prefix}GPU memory usage: {get_gpu_mem():.2f} MB, CPU memory usage: {get_cpu_mem():.2f} MB"


def get_tflops(model_numel, batch_size, seq_len, step_time):
    return model_numel * batch_size * seq_len * 8 / 1e12 / (step_time + 1e-12)


def get_cpu_mem():
    return psutil.Process().memory_info().rss / 1024**2


def get_gpu_mem():
    return torch.cuda.memory_allocated() / 1024**2


## Parameter Sharding Strategies for Tensor Parallelism
def split_param_single_dim_tp1d(dim: int, param: ColoParameter, pg: ProcessGroup):
    spec = (ShardSpec([dim], [pg.tp_world_size()]), ComputeSpec(ComputePattern.TP1D))
    param.set_tensor_spec(*spec)


def split_param_row_tp1d(param: ColoParameter, pg: ProcessGroup):
    split_param_single_dim_tp1d(0, param, pg)


def split_param_col_tp1d(param: ColoParameter, pg: ProcessGroup):
    split_param_single_dim_tp1d(-1, param, pg)


# Tensor Parallel
def tensor_parallelize(model: torch.nn.Module, pg: ProcessGroup):
    """tensor_parallelize
    Sharding the Model Parameters.
    Args:
        model (torch.nn.Module): a torch module to be sharded
    """
    for mn, module in model.named_modules():
        for pn, param in module.named_parameters(recurse=False):
            # NOTE() a param maybe shared by tow modules
            if hasattr(param, "visited"):
                continue
            param.set_dist_spec(ReplicaSpec())
            if "DenseReluDense.w" in mn:
                if "weight" in pn or "bias" in pn:
                    split_param_col_tp1d(param, pg)  # colmn slice
                    # keep the shape of the output from c_fc
                    param.compute_spec.set_output_replicate(False)
                else:
                    param.set_dist_spec(ReplicaSpec())
            elif "SelfAttention" in mn or "EncDecAttention" in mn:
                if "weight" in pn:
                    split_param_row_tp1d(param, pg)  # row slice
                else:
                    param.set_dist_spec(ReplicaSpec())
            elif "lm_head" in mn:
                split_param_col_tp1d(param, pg)  # colmn slice
            elif "SelfAttention" in mn or "c_proj" in mn:
                split_param_col_tp1d(param, pg)  # colmn slice
            else:
                param.set_dist_spec(ReplicaSpec())

            param.visited = True


# Gemini + ZeRO DDP
def gemini_zero_dpp(
    model: torch.nn.Module, pg: ProcessGroup, placememt_policy: str = "auto"
):
    cai_version = colossalai.__version__
    if version.parse(cai_version) > version.parse("0.1.10"):
        from colossalai.nn.parallel import GeminiDDP

        model = GeminiDDP(
            model,
            device=get_current_device(),
            search_range_mb=64,
            placement_policy=placememt_policy,
            pin_memory=True,
        )
    elif version.parse(cai_version) <= version.parse("0.1.10") and version.parse(
        cai_version
    ) >= version.parse("0.1.9"):
        from colossalai.gemini import ChunkManager, GeminiManager

        chunk_size = ChunkManager.search_chunk_size(model, 64 * 1024**2, 32)
        gemini_manager = GeminiManager(placememt_policy, chunk_manager)
        chunk_manager = ChunkManager(
            chunk_size,
            pg,
            enable_distributed_storage=True,
            init_device=GeminiManager.get_default_device(placememt_policy),
        )
        model = ZeroDDP(model, gemini_manager)
    else:
        raise NotImplemented(f"CAI version {cai_version} is not supported")
    return model
