#!/bin/bash

colext_launch_job -p

if [ "$#" -ne 1 ]; then
    echo "Usage: $0 <number_of_background_processes>"
    exit 1
fi

if ! [[ "$1" =~ ^[0-9]+$ ]]; then
    echo "Error: Argument must be an integer."
    exit 1
fi

NUM_BG_PROCS="$1"

sed "s/NUM_BG_PROCESSES/$NUM_BG_PROCS/g" exp/colext/model_profiling_server.yaml | sed "s/USE_CUDA_VAR/\"\"/g" - | sed "s/METANAME/cuda/g" - | mk apply -f -

for device in jao1 jao2 jon1 jon2 jxn2; do
    sed "s/HOSTNAME/$device/g" exp/colext/model_profiling.yaml | sed "s/BASEIMAGE/jetson/g" - | sed "s/NUM_BG_PROCESSES/$NUM_BG_PROCS/g" - | mk apply -f -
done

for device in jn1 jn2; do
    sed "s/HOSTNAME/$device/g" exp/colext/model_profiling.yaml | sed "s/BASEIMAGE/jetson-nano/g" - | sed "s/NUM_BG_PROCESSES/$NUM_BG_PROCS/g" - | mk apply -f -
done

for device in lp1; do
    sed "s/HOSTNAME/$device/g" exp/colext/model_profiling.yaml | sed "s/BASEIMAGE/generic-cpu-x86/g" - | sed "s/NUM_BG_PROCESSES/$NUM_BG_PROCS/g" - | mk apply -f -
done

for device in op1 op2 op3; do
    sed "s/HOSTNAME/$device/g" exp/colext/model_profiling.yaml | sed "s/BASEIMAGE/generic-cpu-arm/g" - | sed "s/NUM_BG_PROCESSES/$NUM_BG_PROCS/g" - | mk apply -f -
done

echo "Waiting for boris-flserver-cuda pod to reach 'Completed' status..."
while true; do
    if mk get pods --no-headers | grep 'boris-flserver-cuda' | awk '{print $3}' | grep -q '^Completed$'; then
        echo "boris-flserver-cuda pod is in 'Completed' status."
        break
    fi
    sleep 10
done

sed "s/NUM_BG_PROCESSES/$NUM_BG_PROCS/g" exp/colext/model_profiling_server.yaml | sed "s/USE_CUDA_VAR/\"0\"/g" - | sed "s/METANAME/cpu/g" - | mk apply -f -

while true; do
    if mk get pods --no-headers | awk '{print $3}' | grep -qv '^Completed$'; then
        echo "Waiting for all pods to reach 'Completed' status..."
        sleep 20
    else
        echo "..."
        break
    fi
done

echo "All pods are in 'Completed' status."

mkdir -p exp/results

FILE="exp/results/device_profiling_${NUM_BG_PROCS}_processes.txt"
if [ -f "$FILE" ]; then
    rm "$FILE"
fi

for pod in $(mk get pods --no-headers | awk '{print $1}'); do
    mk logs "$pod" | grep '^RESULTS' >> "$FILE"
done
