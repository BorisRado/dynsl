#!/bin/bash -l

mamba activate slower

gpu_type=$1

# Assert that $1 is set and its value is a100, v100, or p100
if [[ -z "$gpu_type" || ! "$gpu_type" =~ ^(a100|v100|p100)$ ]]; then
    echo "Error: gpu_type must be one of a100, v100, or p100."
    exit 1
fi

COMMON_ARGS="experiment.num_rounds=5 client.num_iterations=100 model=resnet18 +filepath=times_${gpu_type}"


for device_count in "op:5" "op:40" "op:120"; do

    DEVICE_COUNTS="+experiment.device_counts={$device_count}"
    echo $DEVICE_COUNTS
    # python -u exp/scripts/py/test_policy.py \
    #     algorithm=dynsl_v2 \
    #     algorithm.layer_allocator_init_args.layer_allocation_optimizer=binary_search \
    #     $COMMON_ARGS "${DEVICE_COUNTS}"

    python -u exp/scripts/py/test_policy.py algorithm=splitfed_v2 algorithm.layer_allocator_init_args.num_client_layers=4 $COMMON_ARGS "${DEVICE_COUNTS}"
    python -u exp/scripts/py/test_policy.py algorithm=splitfed_v2 algorithm.layer_allocator_init_args.num_client_layers=5 $COMMON_ARGS "${DEVICE_COUNTS}"
    python -u exp/scripts/py/test_policy.py algorithm=splitfed_v2 algorithm.layer_allocator_init_args.num_client_layers=6 $COMMON_ARGS "${DEVICE_COUNTS}"
    python -u exp/scripts/py/test_policy.py algorithm=splitfed_v2 algorithm.layer_allocator_init_args.num_client_layers=8 $COMMON_ARGS "${DEVICE_COUNTS}"
done


# python -u exp/scripts/py/test_policy.py algorithm=oracle '+algorithm.layer_allocator_init_args.device_mapping={jao:8,jon:8,jxn:8,lp:6,op:6}' $COMMON_ARGS
