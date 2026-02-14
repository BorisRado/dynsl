#!/bin/bash -l


#SBATCH --time=24:00:00
#SBATCH --ntasks=5
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=4GB
#SBATCH --gpus-per-task=1
#SBATCH --out=exp/logs/simulations_separate_heads.log


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

for seed in 4 8 15; do
for dataset in cifar10; do
for model in resnet18; do

COMMON_ARGS=(
    # general configuration
    general.seed=$seed
    general.rounds=$NUM_ROUNDS

    # model configuration
    model=$model
    model.num_groups=16
    model.use_maxpool=false

    # dataset configuration
    data=$dataset
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


# Oralce
# run_bg python exp/scripts/py/simulate_sl.py algorithm=oracle_simulation     \
#     "${COMMON_ARGS[@]}"                                                     \
#     algorithm.num_server_models=-1

# FedAvg
run_bg python exp/scripts/py/simulate_fl.py algorithm=fedavg "${COMMON_ARGS[@]}"


# SplitFedV2
for split_point in 6; do # just do one as an example
    run_bg python exp/scripts/py/simulate_sl.py algorithm=splitfed_v1 \
        "${COMMON_ARGS[@]}" \
        algorithm.layer_allocator_init_args.num_client_layers=$split_point
done

# # DynSL
for exchange_optimizer_states in true false; do
for algorithm in dynsl_simulation dynsl_simulation_one_way; do

    if [[ $exchange_optimizer_states == true && $algorithm == dynsl_simulation_one_way ]]; then
        continue
    fi

    run_bg python exp/scripts/py/simulate_sl.py algorithm=$algorithm \
        "${COMMON_ARGS[@]}" \
        algorithm.exchange_optimizer_states=$exchange_optimizer_states
done
done

done
done
done

wait
