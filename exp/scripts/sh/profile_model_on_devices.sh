#!/bin/bash


# Profile model training time across heterogeneous devices with varying background process loads.
# For each background process count (0, 1, 2)
# 1. trains model on server (w/ and wo/ CUDA) as well as on several edge devices
# 2. collects RESULTS lines from pod logs, and writes to exp/results/profiling_*.txt for analysis.

colext_launch_job -p

MODEL=resnet18

for NUM_BG_PROCS in 0 1 2; do

sed "s/NUM_BG_PROCESSES/$NUM_BG_PROCS/g" exp/colext/model_profiling_server.yaml | sed "s/USE_CUDA_VAR/\"\"/g" - | sed "s/METANAME/cpu/g" - | sed "s/MODEL_NAME/$MODEL/g" - | mk apply -f -

for device in jao1 jao2 jon1 jon2 jxn2; do
    sed "s/HOSTNAME/$device/g" exp/colext/model_profiling.yaml | sed "s/BASEIMAGE/jetson/g" - | sed "s/NUM_BG_PROCESSES/$NUM_BG_PROCS/g" - | sed "s/MODEL_NAME/$MODEL/g" - | mk apply -f -
done

for device in jn1 jn2; do
    sed "s/HOSTNAME/$device/g" exp/colext/model_profiling.yaml | sed "s/BASEIMAGE/jetson-nano/g" - | sed "s/NUM_BG_PROCESSES/$NUM_BG_PROCS/g" - | sed "s/MODEL_NAME/$MODEL/g" - | mk apply -f -
done

for device in lp1; do
    sed "s/HOSTNAME/$device/g" exp/colext/model_profiling.yaml | sed "s/BASEIMAGE/generic-cpu-x86/g" - | sed "s/NUM_BG_PROCESSES/$NUM_BG_PROCS/g" - | sed "s/MODEL_NAME/$MODEL/g" - | mk apply -f -
done

for device in op1 op2 op3; do
    sed "s/HOSTNAME/$device/g" exp/colext/model_profiling.yaml | sed "s/BASEIMAGE/generic-cpu-arm/g" - | sed "s/NUM_BG_PROCESSES/$NUM_BG_PROCS/g" - | sed "s/MODEL_NAME/$MODEL/g" - | mk apply -f -
done

echo "Waiting for boris-flserver-cpu pod to reach 'Completed' status..."
while true; do
    if mk get pods --no-headers | grep 'boris-flserver-cpu' | awk '{print $3}' | grep -q '^Completed$'; then
        echo "boris-flserver-cpu pod is in 'Completed' status."
        break
    fi
    sleep 10
done

sed "s/NUM_BG_PROCESSES/$NUM_BG_PROCS/g" exp/colext/model_profiling_server.yaml | sed "s/USE_CUDA_VAR/\"0\"/g" - | sed "s/METANAME/cuda/g" - | sed "s/MODEL_NAME/$MODEL/g" - | mk apply -f -

while true; do
    if mk get pods --no-headers | awk '{print $3}' | grep -qv '^Completed$'; then
        echo "Waiting for all pods to reach 'Completed' status..."
        sleep 20
    else
        echo "Breaking out..."
        break
    fi
done

echo "All pods are in 'Completed' status."

mkdir -p exp/results

FILE="exp/results/profiling_${NUM_BG_PROCS}procs_${MODEL}.txt"
if [ -f "$FILE" ]; then
    rm "$FILE"
fi

for pod in $(mk get pods --no-headers | awk '{print $1}'); do
    mk logs "$pod" | grep '^RESULTS' >> "$FILE"
done

mk delete pods --all

done
