#!/bin/bash

source ~/.venv/colext/bin/activate

colext_launch_job -p


for MODEL in resnet18 efficientnet mobilenet vit; do

for device in jao1 jao2 jon1 jon2 jxn2; do
    sed "s/HOSTNAME/$device/g" exp/colext/profile_layers.yaml | sed "s/BASEIMAGE/jetson/g" - | sed "s/NUM_BG_PROCESSES/$NUM_BG_PROCS/g" - | sed "s/MODEL_NAME/$MODEL/g" - | mk apply -f -
done

for device in jn1 jn2; do
    sed "s/HOSTNAME/$device/g" exp/colext/profile_layers.yaml | sed "s/BASEIMAGE/jetson-nano/g" - | sed "s/NUM_BG_PROCESSES/$NUM_BG_PROCS/g" - | sed "s/MODEL_NAME/$MODEL/g" - | mk apply -f -
done

for device in lp1; do
    sed "s/HOSTNAME/$device/g" exp/colext/profile_layers.yaml | sed "s/BASEIMAGE/generic-cpu-x86/g" - | sed "s/NUM_BG_PROCESSES/$NUM_BG_PROCS/g" - | sed "s/MODEL_NAME/$MODEL/g" - | mk apply -f -
done

for device in op1 op3; do
    sed "s/HOSTNAME/$device/g" exp/colext/profile_layers.yaml | sed "s/BASEIMAGE/generic-cpu-arm/g" - | sed "s/NUM_BG_PROCESSES/$NUM_BG_PROCS/g" - | sed "s/MODEL_NAME/$MODEL/g" - | mk apply -f -
done

while true; do
    if mk get pods --no-headers | awk '{print $3}' | grep -qv '^Completed$'; then
        echo "Waiting for all pods to reach 'Completed' status..."
        sleep 10
    else
        break
    fi
done

echo "All pods are in 'Completed' status."

rm -f "$MODEL.yaml"

for pod in $(mk get pods --no-headers | awk '{print $1}'); do
    mk logs "$pod" | sed -n '/^========== START OF ACTUAL RESULTS/,/^========== END OF ACTUAL RESULTS/p' | sed '1d;$d' | grep -v '^$' >> "$MODEL.yaml"
done

mk delete pods --all

python exp/scripts/py/average_layer_times.py "$MODEL.yaml"
rm "$MODEL.yaml"

done
