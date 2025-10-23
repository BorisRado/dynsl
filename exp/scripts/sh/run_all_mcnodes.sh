#!/bin/bash -l

#SBATCH --time=4:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=2GB
#SBATCH --gpus-per-task=1
#SBATCH --constraint=gpu_p100
#SBATCH --out=exp/logs/motivation.log

#SBATCH hetjob
#SBATCH --ntasks=8
#SBATCH --cpus-per-task=2
#SBATCH --mem-per-cpu=2GB

#SBATCH hetjob
#SBATCH --ntasks=6
#SBATCH --cpus-per-task=2
#SBATCH --gpus-per-task=1
#SBATCH --mem-per-cpu=2GB


SERVER_IP=$(srun --het-group=0 hostname)
echo "First group hosts: ---${SERVER_IP}---"

export HYDRA_FULL_ERROR=1
export HF_DATASETS_CACHE="/mnt/scratch/radovib"

mamba activate slower

SERVER_PORT=8089
NUM_ROUNDS=8
ALGORITHM=splitfed_v2
BATCH_SIZE=32
SEED=4
NUM_PROCESS_BATCHES=1
SERVER_ADDRESS=${SERVER_IP}:${SERVER_PORT}

NUM_FAST_CLIENTS=6
NUM_SLOW_CLIENTS=8
TOT_NUM_CLIENTS=$((NUM_FAST_CLIENTS + NUM_SLOW_CLIENTS))

RESNET_NUM_PARTITIONS=5
DENSENET_NUM_PARTITIONS=20

echo "Total number of clients: $TOT_NUM_CLIENTS"
echo "Number of fast clients: $NUM_FAST_CLIENTS"
echo "Number of slow clients: $NUM_SLOW_CLIENTS"


start_server() {
    srun \
        --ntasks=1 \
        --het-group=0 \
    python exp/scripts/py/server.py "$@" $GENERAL_SERVER_KWARGS &
}

start_fast_clients() {
    for ((client_idx=0; client_idx<NUM_FAST_CLIENTS; client_idx++)); do
    srun \
        --ntasks=1 \
        --het-group=2 \
        --gpus=1 \
        --output=exp/logs/fast_client_${client_idx}.log \
        --export=ALL,COLEXT_DEVICE_TYPE=WithGPU \
    python exp/scripts/py/client.py "$@" client.server_address=${SERVER_ADDRESS} &
    done
}

start_slow_clients() {
    for ((client_idx=0; client_idx<NUM_SLOW_CLIENTS; client_idx++)); do
    srun \
        --ntasks=1 \
        --het-group=1 \
        --output=exp/logs/client_single_${client_idx}.log \
        --export=ALL,COLEXT_DEVICE_TYPE=WithoutGPU \
    python exp/scripts/py/client.py "$@" client.server_address=${SERVER_ADDRESS} &
    done
}

GENERAL_SERVER_KWARGS="
    server.port=$SERVER_PORT \
    general.rounds=$NUM_ROUNDS \
    wandb=true \
    experiment_name=motivation_experiments \
    server_model_servicer.num_concat_batches=$NUM_PROCESS_BATCHES \
    strategy.min_fit_clients=$TOT_NUM_CLIENTS \
    strategy.min_evaluate_clients=0 \
    strategy.min_available_clients=$TOT_NUM_CLIENTS
"


##################################################################
# RUN THE EXPERIMENT WITH DIFFERENT NUMBER OF CLIENT LAYERS // SPLITFED V2
##################################################################
for model in resnet18 densenet121; do

if [ "$model" == "resnet18" ]; then
    num_partitions=$RESNET_NUM_PARTITIONS
    num_client_layers=(4 5 6 7)
else
    num_client_layers=(4 6 8 10 12)
    num_partitions=$DENSENET_NUM_PARTITIONS
fi

for num_layers in "${num_client_layers[@]}"; do

common_kwargs="
    algorithm=$ALGORITHM
    model=$model
    general.seed=$SEED
    partitioning.num_partitions=$num_partitions
    client.init_kwargs.batch_size=$BATCH_SIZE
"

start_server \
    $common_kwargs \
    algorithm.layer_allocator_init_args.num_client_layers=$num_layers

# wait some time for the server to start up
sleep 10

## START CLIENTS ##

# fast clients
start_fast_clients $common_kwargs
start_slow_clients $common_kwargs

wait # wait for the experiment to finish running

echo "======================================================================="
echo "======================================================================="
echo "======================================================================="

done
done

##################################################################
# REPEAT THE EXPERIMENT WITH AN ORACLE ALLOCATOR
##################################################################

common_kwargs="
    algorithm=oracle
    model=resnet18
    general.seed=$SEED
    partitioning.num_partitions=$RESNET_NUM_PARTITIONS
    client.init_kwargs.batch_size=$BATCH_SIZE
"

start_server \
    $common_kwargs \
    +algorithm.layer_allocator_init_args.device_mapping.WithGPU=7 \
    +algorithm.layer_allocator_init_args.device_mapping.WithoutGPU=5

# fast clients
start_fast_clients $common_kwargs
start_slow_clients $common_kwargs

wait

common_kwargs="
    algorithm=oracle
    model=densenet121
    general.seed=$SEED
    partitioning.num_partitions=$DENSENET_NUM_PARTITIONS
    client.init_kwargs.batch_size=$BATCH_SIZE
"

start_server \
    $common_kwargs \
    +algorithm.layer_allocator_init_args.device_mapping.WithGPU=12 \
    +algorithm.layer_allocator_init_args.device_mapping.WithoutGPU=8

# fast clients
start_fast_clients $common_kwargs
start_slow_clients $common_kwargs

wait

echo "All experiments completed :)"
