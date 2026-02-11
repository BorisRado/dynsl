#!/bin/bash -l

source ~/.venv/colext/bin/activate

NUM_ROUNDS=4
MODEL_NAME="resnet18" # resnet18 / densenet121
EXP_NAME="throughput_test_socket2"
NUM_CLIENT_EPOCHS=2


# NUM_PARTITIONS=$((NUM_JAO + NUM_JN + NUM_JON + NUM_LP + NUM_JXN + NUM_OP))
NUM_PARTITIONS=50
NUM_SERVER_MODELS=-1
NUM_CONCAT_BATCHES=1

for throughput in 25 50 250 500 1000; do

    if [ "$throughput" -ne 1000 ]; then
        ssh -A ubuntu@jxn2 "sudo tc qdisc add dev eth0 root tbf rate ${throughput}mbit burst 32kbit latency 400ms"
    fi

    COMMON_ARGS=(
        # experiment configuration
        --model-name=$MODEL_NAME
        --num-rounds=$NUM_ROUNDS
        --num-partitions=$NUM_PARTITIONS
        --num-server-models=$NUM_SERVER_MODELS
        --num-epochs-per-round=$NUM_CLIENT_EPOCHS
        --num-concat-batches=$NUM_CONCAT_BATCHES
        --sl-communication-channel=socket

        # other configuration
        --experiment-name=$EXP_NAME

        --num-jao=0
        --num-jon=0
        --num-lp=0
        --num-jxn=1
        --num-op=0
        --num-jn=0

        --misc="+stated_throughput=$throughput"
    )


    for num_layers in 4 6 8; do

        python exp/scripts/py/generate_colext_config.py \
            --algorithm=splitfed_v2 \
            --splitfed-num-client-layers=$num_layers \
            "${COMMON_ARGS[@]}"

        colext_launch_job

    done

    if [ "$throughput" -ne 1000 ]; then
        ssh -A ubuntu@jxn2 "sudo tc qdisc del dev eth0 root"
    fi

done
