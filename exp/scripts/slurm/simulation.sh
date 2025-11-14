#!/bin/bash -l

#SBATCH --time=15:00:00
#SBATCH --ntasks=6
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=2GB
#SBATCH --gpus-per-task=1
#SBATCH --out=exp/logs/simulations.log


export HYDRA_FULL_ERROR=1
export HF_DATASETS_CACHE="/mnt/scratch/radovib"

mamba activate slower


BATCH_SIZE=32
NUM_CLIENTS=20
NUM_ROUNDS=200
MODEL=resnet18
INITIAL_LEARNING_RATE=0.1
LR_SCHEDULING=cosine

for seed in 8 15; do

COMMON_ARGS="
    general.seed=$seed
    model=$MODEL
    general.rounds=$NUM_ROUNDS
    partitioning.num_partitions=${NUM_CLIENTS}
    client.init_kwargs.batch_size=$BATCH_SIZE
    optimizer.optimizer.lr=$INITIAL_LEARNING_RATE
    optimizer/scheduler=$LR_SCHEDULING
    +simulation=true
    algorithm.num_server_models=-1
    wandb=true
    experiment_name=simulation3
"

srun --ntasks=1 --quiet --nodes=1 python exp/scripts/py/simulate_sl.py algorithm=dynsl_simulation $COMMON_ARGS &
srun --ntasks=1 --quiet --nodes=1 python exp/scripts/py/simulate_sl.py algorithm=fedavg $COMMON_ARGS &
srun --ntasks=1 --quiet --nodes=1 python exp/scripts/py/simulate_sl.py algorithm=oracle_simulation $COMMON_ARGS &

srun --ntasks=1 --quiet --nodes=1 python exp/scripts/py/simulate_sl.py algorithm=splitfed_v1 $COMMON_ARGS \
    algorithm.layer_allocator_init_args.num_client_layers=4 &
srun --ntasks=1 --quiet --nodes=1 python exp/scripts/py/simulate_sl.py algorithm=splitfed_v1 $COMMON_ARGS \
    algorithm.layer_allocator_init_args.num_client_layers=6 &

done

wait
