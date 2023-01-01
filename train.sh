# The following options only valid when DISTPAN="colossalai"
export TPDEGREE=1
export GPUNUM=1
export PLACEMENT='cpu'
export USE_SHARD_INIT=False

env OMP_NUM_THREADS=40 colossalai run  --nproc_per_node=1 \
    colossalai_baseline.py \
    --tp_degree=${TPDEGREE} \
    --placement ${PLACEMENT} \
    --shardinit ${USE_SHARD_INIT}
