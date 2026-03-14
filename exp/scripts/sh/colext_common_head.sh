#!/bin/bash -l

source ~/.venv/colext/bin/activate


NUM_ROUNDS=200
EXP_NAME="colext_common_head_constant_lr"

NUM_JAO="2"
NUM_JXN="2"
NUM_JON="4"
NUM_JN="6"
NUM_LP="4"
NUM_OP="7"


NUM_PARTITIONS=$((NUM_JAO + NUM_JN + NUM_JON + NUM_LP + NUM_JXN + NUM_OP))

num_client_epochs=1
num_concat_batches=1

common_args=(

    # algorithm arguments that alter algorithm behavior
    --num-concat-batches=$num_concat_batches
    --num-epochs-per-round=$num_client_epochs
    --use-groupnorm=false

    --optimizer=adam

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

)

for model in resnet18 efficientnet; do

    if [[ "$model" == "resnet18" ]]; then
        split_points=(5 6 7)
    elif [[ "$model" == "efficientnet" ]]; then
        split_points=(3 6 8)
    elif [[ "$model" == "mobilenet" ]]; then
        split_points=(5 10 15)
    elif [[ "$model" == "vit" ]]; then
        split_points=(2 3 5)
    else
        echo "Unknown model: $model"
        exit 0
    fi

    for split_point in "${split_points[@]:0:2}"; do
        python exp/scripts/py/generate_colext_config.py \
            --algorithm=splitfed_v2 \
            --model-name=$model \
            --splitfed-num-client-layers=$split_point \
            --num-server-models=1 \
            "${common_args[@]}"
        colext_launch_job

    done

    python exp/scripts/py/generate_colext_config.py \
        --algorithm=dynsl_w_profiling \
        --model-name=$model \
        --num-server-models=1 \
        "${common_args[@]}"
    colext_launch_job


    python exp/scripts/py/generate_colext_config.py \
        --algorithm=dynsl_v2 \
        --model-name=$model \
        --dynsl-exchange-optimizer-states=false \
        --num-server-models=1 \
        "${common_args[@]}"
    colext_launch_job


    python exp/scripts/py/generate_colext_config.py \
        --algorithm=dynsl_w_profiling \
        --model-name=$model \
        --num-server-models=-1 \
        "${common_args[@]}"
    colext_launch_job

    for split_point in "${split_points[@]:0:2}"; do
        python exp/scripts/py/generate_colext_config.py \
            --algorithm=sfsl \
            --model-name=$model \
            --num-server-models=1 \
            --sfsl-num-client-layers=$split_point \
            "${common_args[@]}"
        colext_launch_job

    done
done
