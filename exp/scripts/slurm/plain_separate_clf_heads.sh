#!/bin/bash -l


#SBATCH --time=24:00:00
#SBATCH --ntasks=5
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=4GB
#SBATCH --gpus-per-task=1
#SBATCH --out=exp/logs/simulations_separate_heads.log


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
# run_bg python exp/scripts/py/simulate_sl.py algorithm=oracle_simulation     \
#     "${COMMON_ARGS[@]}"                                                     \
#     algorithm.num_server_models=-1

# FedAvg
run_bg python exp/scripts/py/simulate_fl.py algorithm=fedavg "${EXP_ARGS[@]}"


# SplitFedV1
for split_point in 6; do # just do one as an example
    run_bg python exp/scripts/py/simulate_sl.py algorithm=splitfed_v1 \
        "${EXP_ARGS[@]}" \
        algorithm.layer_allocator_init_args.num_client_layers=$split_point
done
# SplitFedV1 -- END

# DynSL
for exchange_optimizer_states in true false; do
for algorithm in dynsl_simulation dynsl_simulation_one_way; do

    if [[ $exchange_optimizer_states == true && $algorithm == dynsl_simulation_one_way ]]; then
        continue
    fi

    run_bg python exp/scripts/py/simulate_sl.py algorithm=$algorithm \
        "${EXP_ARGS[@]}" \
        algorithm.exchange_optimizer_states=$exchange_optimizer_states
done
done
# DynSL -- END

done
done
done

wait
