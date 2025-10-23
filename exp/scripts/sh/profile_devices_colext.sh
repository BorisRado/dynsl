#!/bin/bash

if [ "$#" -ne 1 ]; then
    echo "Usage: $0 <number_of_background_processes>"
    exit 1
fi

if ! [[ "$1" =~ ^[0-9]+$ ]]; then
    echo "Error: Argument must be an integer."
    exit 1
fi

NUM_BG_PROCS="$1"

for device in jao1 jao2 jon1 jon2 jxn1; do
    sed "s/HOSTNAME/$device/g" exp/colext/model_profiling.yaml | sed "s/BASEIMAGE/jetson/g" - | sed "s/NUM_BG_PROCESSES/$NUM_BG_PROCS/g" - | mk apply -f -
done

for device in jn1 jn2; do
    sed "s/HOSTNAME/$device/g" exp/colext/model_profiling.yaml | sed "s/BASEIMAGE/jetson-nano/g" - | sed "s/NUM_BG_PROCESSES/$NUM_BG_PROCS/g" - | mk apply -f -
done

for device in lp1 lp5; do
    sed "s/HOSTNAME/$device/g" exp/colext/model_profiling.yaml | sed "s/BASEIMAGE/generic-cpu-x86/g" - | sed "s/NUM_BG_PROCESSES/$NUM_BG_PROCS/g" - | mk apply -f -
done

for device in op1 op2 op3; do
    sed "s/HOSTNAME/$device/g" exp/colext/model_profiling.yaml | sed "s/BASEIMAGE/generic-cpu-arm/g" - | sed "s/NUM_BG_PROCESSES/$NUM_BG_PROCS/g" - | mk apply -f -
done

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

FILE="device_profiling_${NUM_BG_PROCS}_processes.txt"
if [ -f "$FILE" ]; then
    rm "$FILE"
fi

for pod in $(mk get pods --no-headers | awk '{print $1}'); do
    mk logs "$pod" | grep '^RESULTS' >> "$FILE"
done
