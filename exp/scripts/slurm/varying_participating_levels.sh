#!/bin/bash -l

#SBATCH --time=1-00:00:00
#SBATCH --ntasks=4
#SBATCH --cpus-per-task=16
#SBATCH --mem-per-cpu=4GB
#SBATCH --gpus-per-task=1
#SBATCH --constraint="gpu_v100&cpu_intel_xeon_silver_4112"
#SBATCH --out=exp/logs/varying_participating_levels.log

srun true # NECESSARY TO AVOID SLURM ISSUES!!!!

mamba activate slower

run_bg() {
    srun -Q -N1 --ntasks=1 "$@" &
}

MODEL=efficientnet
COMMON_ARGS="experiment.num_rounds=3 client.num_iterations=150 model=$MODEL +filepath=_auto_ algorithm.num_server_models=1"

if [[ "$MODEL" == "resnet18" ]]; then
    split_points=(5 6 7)
elif [[ "$MODEL" == "mobilenet" ]]; then
    split_points=(5 10 15)
elif [[ "$MODEL" == "efficientnet" ]]; then
    split_points=(3 5 7)
else
    echo "Unknown MODEL: $MODEL"
    exit 0
fi


for device_count in \
    "lp:10" \
    "op:10" \
    "jn:10" \
    "jn:10,op:10" \
    "jn:10,lp:10" \
    "op:10,lp:10" \
    "lp:50,op:10" \
    "lp:10,op:50" \
    "lp:50,op:50" \
    "lp:10,jn:50" \
    "lp:50,jn:10" \
    "op:10,jn:50" \
    "op:10,lp:10,jn:10" \
    "op:10,lp:10,jn:50" \
    "op:10,lp:50,jn:10" \
    "op:50,lp:50,jn:50"
do
    DEVICE_COUNTS="+experiment.device_counts={$device_count}"
    echo $DEVICE_COUNTS

    run_bg python -u exp/scripts/py/test_policy.py algorithm=dynsl_v2 $COMMON_ARGS "${DEVICE_COUNTS}" algorithm.server_optim_state_handling=ignore algorithm.layer_allocator_init_args.reallocation_frequency=4.0
    run_bg python -u exp/scripts/py/test_policy.py algorithm=dynsl_w_profiling $COMMON_ARGS "${DEVICE_COUNTS}"
    run_bg python -u exp/scripts/py/test_policy.py algorithm=ares $COMMON_ARGS "${DEVICE_COUNTS}"

    for layer in "${split_points[@]}"; do
        run_bg python -u exp/scripts/py/test_policy.py \
            algorithm=splitfed_v2 \
            algorithm.layer_allocator_init_args.num_client_layers=$layer \
            $COMMON_ARGS \
            "${DEVICE_COUNTS}"

    done

done

wait
