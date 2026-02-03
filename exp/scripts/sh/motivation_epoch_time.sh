#!/bin/bash -l

source ~/.venv/colext/bin/activate

NUM_ROUNDS=6
MODEL_NAME="densenet121" # resnet18 / densenet121
EXP_NAME="motivation_epoch_time"
NUM_CLIENT_EPOCHS=5

NUM_JAO="2"
NUM_JXN="1"
NUM_JON="4"

NUM_JN="6"
NUM_LP="4"
NUM_OP="7"

NUM_PARTITIONS=$((NUM_JAO + NUM_JN + NUM_JON + NUM_LP + NUM_JXN + NUM_OP))

# NOTE: The oracle configurations have been obtained via grid search.
# They may not be optimal if using a different number of clients

if [ "$MODEL_NAME" == "densenet121" ]; then
    ORACLE_CONFIGURATION=(
        --oracle-nl-jao=16
        --oracle-nl-jxn=16
        --oracle-nl-jon=16
        --oracle-nl-jn=8
        --oracle-nl-op=4
        --oracle-nl-lp=4
    )
    TEST_SPLITFED_CLIENT_LAYERS=(4 6 8 10 11) # avoid 12 as it is too slow
    TEST_FEDAVG=false # too slow for densenet121
elif [ "$MODEL_NAME" == "resnet18" ]; then
    ORACLE_CONFIGURATION=(
        --oracle-nl-jxn=11
        --oracle-nl-jao=11
        --oracle-nl-jon=11
        --oracle-nl-jn=11
        --oracle-nl-op=6
        --oracle-nl-lp=4
    )
    TEST_SPLITFED_CLIENT_LAYERS=(4 5 6 8)
    TEST_FEDAVG=true
else
    echo "Unsupported model: $MODEL_NAME"
    exit 1
fi

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

    # number of participating devices
    --num-jao=$NUM_JAO
    --num-jon=$NUM_JON
    --num-jn=$NUM_JN
    --num-lp=$NUM_LP
    --num-jxn=$NUM_JXN
    --num-op=$NUM_OP

    # other configuration
    --experiment-name=$EXP_NAME
)

if [ "$TEST_FEDAVG" = true ]; then
    python exp/scripts/py/generate_colext_config.py \
        --algorithm=fedavg \
        "${COMMON_ARGS[@]}"
    colext_launch_job
    sleep 180
fi

## SPLITFED
for num_client_layers in "${TEST_SPLITFED_CLIENT_LAYERS[@]}"; do
python exp/scripts/py/generate_colext_config.py \
    --algorithm=splitfed_v2 \
    --splitfed-num-client-layers=$num_client_layers \
    "${COMMON_ARGS[@]}"
colext_launch_job

sleep 180

done

## ORACLE
python exp/scripts/py/generate_colext_config.py \
    --algorithm=oracle \
    "${COMMON_ARGS[@]}" \
    "${ORACLE_CONFIGURATION[@]}"
colext_launch_job

sleep 180
