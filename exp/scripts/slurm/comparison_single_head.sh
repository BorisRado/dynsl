#!/bin/bash -l

#SBATCH --time=1-00:00:00
#SBATCH --ntasks=5
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=4GB
#SBATCH --gpus-per-task=1
#SBATCH --out=exp/logs/all_constant.log

srun true # NECESSARY TO AVOID SLURM ISSUES!!!!

export HYDRA_FULL_ERROR=1
export HF_DATASETS_CACHE="/mnt/scratch/radovib"

mamba activate slower

# Helper function to run commands asynchronously with srun
run_bg() {
    srun -Q -N1 --ntasks=1 "$@" &
}

BATCH_SIZE=64
NUM_CLIENT_EPOCHS=1
NUM_CLIENTS=40
NUM_ROUNDS=300 # either way there is early stopping
OPTIMIZER=adam
LR_SCHEDULING=constant
INITIAL_LEARNING_RATE=0.001

COMMON_ARGS=(
    # general configuration
    general.rounds=$NUM_ROUNDS
    general.patience=12
    +general.training_speed_imbalance_value=1.0
    model.use_groupnorm=false

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
    optimizer.optimizer.weight_decay=0.0005

    # wandb tracking
    wandb=true
    experiment_name=simulation_complete_resnet18_new
)

model=resnet18

for seed in 4 8 15 16 23; do
for dataset in cifar10 cifar100 cinic; do

COMMON_EXP_ARGS=(
    general.seed=$seed
    model=$model
    data=$dataset
    ${COMMON_ARGS[@]}
)

split_points=(5 6 7)
split_points_pylist="[5,6,7]"
compute_configuration='[{ids:"0-13",num_layers:5},{ids:"14-27",num_layers:6},{ids:"28-40",num_layers:7}]'

run_bg python exp/scripts/py/simulate_sl.py algorithm=splitfed_v1 \
    "${COMMON_EXP_ARGS[@]}" \
    algorithm.layer_allocator_init_args.num_client_layers=5

# SplitFedV2 -- need to compare different split points
for split_point in "${split_points[@]}"; do
    run_bg python exp/scripts/py/simulate_sl.py algorithm=splitfed_v2 \
        "${COMMON_EXP_ARGS[@]}" \
        algorithm.num_server_models=1 \
        algorithm.layer_allocator_init_args.num_client_layers=$split_point
done

run_bg python exp/scripts/py/simulate_sl.py algorithm=splitfed_v2 \
        "${COMMON_EXP_ARGS[@]}" \
        algorithm.num_server_models=1 \
        algorithm.layer_allocator_init_args.num_client_layers=$split_point

# Oracle
run_bg python exp/scripts/py/simulate_sl.py algorithm=oracle_simulation \
    "${COMMON_EXP_ARGS[@]}" \
    algorithm.num_server_models=1 \
    "algorithm.layer_allocator_init_args.client_configuration=$compute_configuration"


for split_point in "${split_points[@]:0:2}"; do
run_bg python exp/scripts/py/simulate_sl.py algorithm=sfsl \
    "${COMMON_EXP_ARGS[@]}" \
    algorithm.num_server_models=1 \
    algorithm.num_client_layers=$split_point
done

# DTFL
run_bg python exp/scripts/py/simulate_sl.py algorithm=dtfl_simulation \
    "${COMMON_EXP_ARGS[@]}" \
    algorithm.num_server_models=1 \
    algorithm.strategy_init_args.all_split_points=$split_points_pylist \
    "algorithm.layer_allocator_init_args.client_configuration=$compute_configuration"


# DynSL -- consider only common model on the server
for weight_integration_strategy in reset_optim_state; do #  ignore keep_optim_state
    run_bg python exp/scripts/py/simulate_sl.py algorithm=dynsl_simulation \
        "${COMMON_EXP_ARGS[@]}" \
        algorithm.num_server_models=1 \
        algorithm.server_optim_state_handling=$weight_integration_strategy \
        algorithm.exchange_optimizer_states=false
done

run_bg python exp/scripts/py/simulate_sl.py algorithm=dynsl_simulation_one_way \
    "${COMMON_EXP_ARGS[@]}" \
    algorithm.num_server_models=1 \
    algorithm.server_optim_state_handling=ignore \
    algorithm.exchange_optimizer_states=false



done
done

wait
