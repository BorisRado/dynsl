#!/bin/bash -l

#SBATCH --time=3-00:00:00
#SBATCH --ntasks=14
#SBATCH --cpus-per-task=10
#SBATCH --mem-per-cpu=4GB
#SBATCH --gpus-per-task=1
#SBATCH --out=exp/logs/all.log

srun true # NECESSARY TO AVOID SLURM ISSUES!!!!

export HYDRA_FULL_ERROR=1
export HF_DATASETS_CACHE="/mnt/scratch/radovib"

mamba activate slower

BATCH_SIZE=48
NUM_CLIENT_EPOCHS=1
NUM_CLIENTS=40
NUM_ROUNDS=300 # either way there is early stopping
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
    general.patience=12

    # model configuration
    # model.num_groups=16

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
    optimizer.optimizer.weight_decay=0.0002

    # wandb tracking
    wandb=true
    experiment_name=simulation_complete2
)


for seed in 4 8 15; do
for dataset in cifar10 cifar100 cinic; do # cifar10 cifar100
for model in efficientnet; do # resnet18 resnet34 efficientnet densenet

COMMON_EXP_ARGS=(
    general.seed=$seed
    model=$model
    data=$dataset
    ${COMMON_ARGS[@]}
)

if [[ "$model" == "resnet18" ]]; then
    split_points=(4 6 7)
    compute_configuration='[{ids:"0-13",num_layers:4},{ids:"14-27",num_layers:6},{ids:"28-40",num_layers:7}]'
    dynsl_file="dynsl_simulation"
elif [[ "$model" == "densenet" ]]; then
    split_points=(3 5 9)
    compute_configuration='[{ids:"0-13",num_layers:3},{ids:"14-27",num_layers:5},{ids:"28-40",num_layers:9}]'
    dynsl_file="dynsl_simulation_densenet"
elif [[ "$model" == "efficientnet" ]]; then
    split_points=(2 5 8)
    compute_configuration='[{ids:"0-13",num_layers:2},{ids:"14-27",num_layers:5},{ids:"28-40",num_layers:8}]'
    dynsl_file="dynsl_simulation_efficientnet"
else
    echo "Unknown model: $model"
    exit 0
fi

# SplitFedV2 -- need to compare different split points
for split_point in "${split_points[@]}"; do
    run_bg python exp/scripts/py/simulate_sl.py algorithm=splitfed_v2 \
        "${COMMON_EXP_ARGS[@]}" \
        algorithm.layer_allocator_init_args.num_client_layers=$split_point
done

# Oracle -- need to compare single classificaiton head with separate classification head per split point
for num_server_models in 1 0; do
    run_bg python exp/scripts/py/simulate_sl.py algorithm=oracle_simulation \
        "${COMMON_EXP_ARGS[@]}" \
        algorithm.num_server_models=$num_server_models \
        "algorithm.layer_allocator_init_args.client_configuration=$compute_configuration"
done

# SFSL -- consider only separate classification head per client as in the original paper
for split_point in "${split_points[@]}"; do
run_bg python exp/scripts/py/simulate_sl.py algorithm=sfsl \
    "${COMMON_EXP_ARGS[@]}" \
    algorithm.num_server_models=1 \
    algorithm.num_client_layers=$split_point
done


# DynSL -- consider only common model on the server
run_bg python exp/scripts/py/simulate_sl.py algorithm=$dynsl_file \
    "${COMMON_EXP_ARGS[@]}" \
    algorithm.num_server_models=1 \
    algorithm.exchange_optimizer_states=false


# DTFL -- only consider separate clf head per client as in the original paper
run_bg python exp/scripts/py/simulate_sl.py algorithm=dtfl_simulation \
    "${COMMON_EXP_ARGS[@]}" \
    algorithm.num_server_models=1 \
    algorithm.strategy_init_args.all_split_points=[3,5,9] \
    "algorithm.layer_allocator_init_args.client_configuration=$compute_configuration"


done
done
done

wait
