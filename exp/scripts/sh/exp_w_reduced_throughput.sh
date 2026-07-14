#!/bin/bash -l

source ~/.venv/colext/bin/activate


NUM_ROUNDS=4
EXP_NAME="reduced_throughput_testing2"

NUM_JAO="2"
NUM_JXN="2"
NUM_JON="3"
NUM_JN="6"
NUM_LP="3"
NUM_OP="8"


NUM_PARTITIONS=$((NUM_JAO + NUM_JN + NUM_JON + NUM_LP + NUM_JXN + NUM_OP))

num_server_models=-1
num_client_epochs=1
num_concat_batches=1

common_args=(

    # algorithm arguments that alter algorithm behavior
    --num-concat-batches=$num_concat_batches
    --num-server-models=$num_server_models
    --num-epochs-per-round=$num_client_epochs

    # general arguments
    --num-rounds=$NUM_ROUNDS
    --num-partitions=$NUM_PARTITIONS
    --experiment-name=$EXP_NAME

    # set the number of devices
    --num-jao=$NUM_JAO
    --num-jon=$NUM_JON
    --num-jn=$NUM_JN
    --num-lp=$NUM_LP
    --num-jxn=$NUM_JXN
    --num-op=$NUM_OP

    --optimizer=sgd
)

model=resnet18

for reduced_throughput in 10 20 50 100 -1 250; do

    throughput_args="--misc=+client.init_kwargs.reduced_network_throughput=${reduced_throughput}"
    if [[ "$model" == "resnet18" ]]; then
        split_points=(5 6 7)
    elif [[ "$model" == "efficientnet" ]]; then
        split_points=(3 6 8)
    elif [[ "$model" == "mobilenet" ]]; then
        split_points=(5 10 15)
    elif [[ "$model" == "densenet" ]]; then
        split_points=(2 4 6)
    elif [[ "$model" == "vit" ]]; then
        split_points=(2 3 5)
    else
        echo "Unknown model: $model"
        exit 0
    fi

    # SPLITFED V1
    for split_point in "${split_points[@]}"; do
        python exp/scripts/py/generate_colext_config.py \
            --algorithm=splitfed_v1 \
            --model-name=$model \
            --splitfed-num-client-layers=$split_point \
            "${common_args[@]}" $throughput_args
        colext_launch_job
    done

    # DynSL with profiling
    python exp/scripts/py/generate_colext_config.py \
        --algorithm=dynsl_w_profiling \
        --num-server-models=-1 \
        --model-name=$model \
        "${common_args[@]}" $throughput_args
    colext_launch_job

    # DynSL
    python exp/scripts/py/generate_colext_config.py \
        --algorithm=dynsl_v2 \
        --num-server-models=-1 \
        --model-name=$model \
        --dynsl-exchange-optimizer-states=true \
        "${common_args[@]}" $throughput_args
    colext_launch_job

done
