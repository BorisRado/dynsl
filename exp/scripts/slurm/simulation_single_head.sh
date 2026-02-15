#!/bin/bash -l


#SBATCH --time=24:00:00
#SBATCH --ntasks=6
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=4GB
#SBATCH --gpus-per-task=1
#SBATCH --out=exp/logs/simulations_single_head.log

source exp/scripts/slurm/simulation_common.sh

for seed in 4 8 15; do
for dataset in cifar10; do
for model in resnet18; do

EXP_ARGS=(
    general.seed=$seed
    model=$model
    data=$dataset
    ${COMMON_ARGS[@]}
)


# Oralce
run_bg python exp/scripts/py/simulate_sl.py algorithm=oracle_simulation     \
    "${EXP_ARGS[@]}"                                                     \
    algorithm.num_server_models=1

# SplitFedV2
for split_point in 4 6 8; do # just do one as an example
    run_bg python exp/scripts/py/simulate_sl.py algorithm=splitfed_v2 \
        "${EXP_ARGS[@]}" \
        algorithm.layer_allocator_init_args.num_client_layers=$split_point
done

# DynSL
for algorithm in dynsl_simulation dynsl_simulation_one_way; do
for exchange_optimizer_states in true false; do
    if [[ $exchange_optimizer_states == true && $algorithm == dynsl_simulation_one_way ]]; then
        continue
    fi

    # server may send optimizer states to the client when offloading layers to the client, but server
    # optimizer is always re-initialized -- never send optimizer states to the server
    run_bg python exp/scripts/py/simulate_sl.py algorithm=$algorithm \
        "${EXP_ARGS[@]}" \
        algorithm.num_server_models=1 \
        algorithm.exchange_optimizer_states=$exchange_optimizer_states \
        algorithm.client_adapter_args.exchange_optimizer_states=false
done
done

done
done
done

wait
