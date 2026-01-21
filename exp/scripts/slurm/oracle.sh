#!/bin/bash -l

#SBATCH --time=36:00:00
#SBATCH --ntasks=4
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=4G
#SBATCH --gpus-per-task=1
#SBATCH --out=exp/logs/oracle.log


export HYDRA_FULL_ERROR=1
export HF_DATASETS_CACHE="/mnt/scratch/radovib"

mamba activate slower


BATCH_SIZE=32
NUM_CLIENTS=20
NUM_ROUNDS=200
MODEL=resnet18
OPTIMIZER=adam
INITIAL_LEARNING_RATE=0.001


if [ "$MODEL" = "vgg9" ]; then
    echo "Setting vgg9 model args"
    MODEL_ARGS="
        model=$MODEL
        algorithm.layer_allocator_init_args.client_configuration.0.num_layers=3
        algorithm.layer_allocator_init_args.client_configuration.1.num_layers=6
        algorithm.layer_allocator_init_args.client_configuration.2.num_layers=9
    "
elif [ "$MODEL" = "resnet18" ]; then
    echo "Setting resnet18 model args"
    MODEL_ARGS="
        model=$MODEL
        model.use_groupnorm=true
        algorithm.layer_allocator_init_args.client_configuration.0.num_layers=4
        algorithm.layer_allocator_init_args.client_configuration.1.num_layers=6
        algorithm.layer_allocator_init_args.client_configuration.2.num_layers=8
    "
fi

# Helper function to run commands asynchronously with srun
run_bg() {
    srun -Q --ntasks=1 "$@" &
}

for seed in 4 8 15; do
for num_concat_batches in 1 2; do
for num_server_models in 1 -1; do
for lr_scheduling in cosine constant; do

if [ $num_concat_batches -eq 2 ] && [ $num_server_models -eq -1 ]; then
    continue
fi


# Test with separate model per split point
run_bg python -u exp/scripts/py/simulate_sl.py algorithm=oracle_simulation  \
    general.seed=$seed                                                      \
    $MODEL_ARGS                                                             \
    general.rounds=$NUM_ROUNDS                                              \
    partitioning.num_partitions=${NUM_CLIENTS}                              \
    client.init_kwargs.batch_size=$BATCH_SIZE                               \
    optimizer.optimizer.lr=$INITIAL_LEARNING_RATE                           \
    optimizer/scheduler=$lr_scheduling                                      \
    optimizer.scheduler.T_max=$NUM_ROUNDS                                   \
    optimizer=$OPTIMIZER                                                    \
    wandb=true                                                              \
    server_model_servicer.num_concat_batches=$num_concat_batches            \
    experiment_name=simulation_oracle3                                      \
    algorithm.num_server_models=$num_server_models

done
done
done
done
wait
