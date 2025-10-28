#!/bin/bash -l

source ~/.venv/colext/bin/activate

MODEL_NAME="resnet18"
NUM_ROUNDS=8
NUM_PARTITIONS=5

COMMON_ARGS="
    --model-name=$MODEL_NAME
    --num-rounds=$NUM_ROUNDS
    --num-partitions=$NUM_PARTITIONS
    --experiment-name=colext2
"

## DYNSL
for cache_client_configuration in true false; do
python  exp/scripts/py/generate_colext_config.py \
    --algorithm=dynsl_v2 \
    $COMMON_ARGS \
    --dynsl-cache-client-configuration=$cache_client_configuration \
    --dynsl-gamma=6.0 \
    --dynsl-reallocation-frequency=5.0
colext_launch_job
sleep 60
done

## SPLITFED
for value in 4 5 6 8; do
    python exp/scripts/py/generate_colext_config.py \
        --algorithm=splitfed_v2 \
        --splitfed-num-client-layers=$value \
        $COMMON_ARGS
    colext_launch_job

    sleep 60
done

## ORACLE
python exp/scripts/py/generate_colext_config.py \
    --algorithm=oracle $COMMON_ARGS \
    --oracle-nl-jao=8 \
    --oracle-nl-jxn=8 \
    --oracle-nl-jon=8 \
    --oracle-nl-jn=6 \
    --oracle-nl-op=6 \
    --oracle-nl-lp=6
colext_launch_job
