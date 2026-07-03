#!/bin/bash -l

source ~/.venv/colext/bin/activate

colext_launch_job -p

NUM_ROUNDS=4
MODEL_NAME="resnet18" # resnet18
EXP_NAME="throughput_test_socket8"
NUM_CLIENT_EPOCHS=2
NUM_PARTITIONS=50
NUM_SERVER_MODELS=-1
NUM_CONCAT_BATCHES=1

COMMON_ARGS=(
    # experiment configuration
    --model-name=$MODEL_NAME
    --num-rounds=$NUM_ROUNDS
    --num-partitions=$NUM_PARTITIONS
    --num-server-models=$NUM_SERVER_MODELS
    --num-epochs-per-round=$NUM_CLIENT_EPOCHS
    --num-concat-batches=$NUM_CONCAT_BATCHES

    # other configuration
    --experiment-name=$EXP_NAME

    --num-jao=0
    --num-jon=0
    --num-lp=0
    --num-jxn=1
    --num-op=0
    --num-jn=0
)

for throughput in 25 50 250 500 1000; do

    if [ "$throughput" -ne 1000 ]; then
        # not sure which JXN is chosen, so we reduce throughput on both
        ssh -A ubuntu@jxn1 "sudo tc qdisc add dev eth0 root tbf rate ${throughput}mbit burst 32kbit latency 400ms"
        ssh -A ubuntu@jxn2 "sudo tc qdisc add dev eth0 root tbf rate ${throughput}mbit burst 32kbit latency 400ms"
    fi

    for num_layers in 5 6 8; do

        python exp/scripts/py/generate_colext_config.py \
            --algorithm=splitfed_v2 \
            --splitfed-num-client-layers=$num_layers \
            --misc="+stated_throughput=$throughput" \
            "${COMMON_ARGS[@]}"

        colext_launch_job

    done

    if [ "$throughput" -ne 1000 ]; then
        ssh -A ubuntu@jxn1 "sudo tc qdisc del dev eth0 root"
        ssh -A ubuntu@jxn2 "sudo tc qdisc del dev eth0 root"
    fi

done
