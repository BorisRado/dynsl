#!/bin/bash -l

#SBATCH --time=18:00:00
#SBATCH --ntasks=8
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=2GB
#SBATCH --gpus-per-task=1
#SBATCH --out=exp/logs/oracle.log


export HYDRA_FULL_ERROR=1
export HF_DATASETS_CACHE="/mnt/scratch/radovib"

mamba activate slower


BATCH_SIZE=32
NUM_CLIENTS=20
NUM_ROUNDS=150
MODEL=vgg9
OPTIMIZER=adam
LR_SCHEDULING=cosine
INITIAL_LEARNING_RATE=0.001

# Helper function to run commands asynchronously with srun
run_bg() {
    srun -Q --ntasks=1 "$@" &
}

for seed in 4 8 15; do
for num_concat_batches in 1 2; do
for num_server_models in 0 1 -1; do
for model in vgg9; do
for lr_scheduling in cosine constant; do

if [ $num_concat_batches -eq 2 ] && [ $num_server_models -eq -1 ]; then
    continue
fi


# Test with separate model per split point
run_bg python exp/scripts/py/simulate_sl.py algorithm=oracle_simulation     \
    general.seed=$seed                                                      \
    model=$model                                                            \
    general.rounds=$NUM_ROUNDS                                              \
    partitioning.num_partitions=${NUM_CLIENTS}                              \
    client.init_kwargs.batch_size=$BATCH_SIZE                               \
    optimizer.optimizer.lr=$INITIAL_LEARNING_RATE                           \
    optimizer/scheduler=$lr_scheduling                                      \
    optimizer.scheduler.T_max=$NUM_ROUNDS                                   \
    optimizer=$OPTIMIZER                                                    \
    wandb=true                                                              \
    server_model_servicer.num_concat_batches=$num_concat_batches            \
    experiment_name=simulation_oracle2                                      \
    algorithm.num_server_models=$num_server_models

done
done
done
done
done
wait
