#!/bin/bash -l

srun true # NECESSARY TO AVOID SLURM ISSUES!!!!

export HYDRA_FULL_ERROR=1
export HF_DATASETS_CACHE="/mnt/scratch/radovib"

mamba activate slower


BATCH_SIZE=32
NUM_CLIENT_EPOCHS=1
NUM_CLIENTS=40
NUM_ROUNDS=250
OPTIMIZER=adam
LR_SCHEDULING=constant
INITIAL_LEARNING_RATE=0.001

# Helper function to run commands asynchronously with srun
run_bg() {
    srun -Q -N1 --ntasks=1 "$@" &
}

COMMON_ARGS=(
    # general configuration
    general.rounds=$NUM_ROUNDS

    # model configuration
    model.num_groups=16
    model.use_maxpool=false

    # dataset configuration
    data.load_onto_memory=true
    partitioning.num_partitions=$NUM_CLIENTS

    # client configuration
    client.init_kwargs.num_epochs=$NUM_CLIENT_EPOCHS
    client.init_kwargs.batch_size=$BATCH_SIZE

    # optimizer configuration
    optimizer=$OPTIMIZER
    optimizer.optimizer.lr=$INITIAL_LEARNING_RATE
    optimizer/scheduler=$LR_SCHEDULING

    # wandb tracking
    wandb=true
    experiment_name=simulation_v2
)
