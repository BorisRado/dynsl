#!/bin/bash -l

#SBATCH --time=4:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=1GB
#SBATCH --gpus=1
#SBATCH --constraint=gpu_p100
#SBATCH --out=exp/logs/test_size.log

#SBATCH hetjob
#SBATCH --ntasks=14
#SBATCH --cpus-per-task=2
#SBATCH --mem-per-cpu=1GB

srun --het-group=0 nvidia-smi

server_ip=$(srun --het-group=0 hostname)
echo "First group hosts: ---${server_ip}---"

mamba activate slower

# for now, simple test with one client and one server to check that everything works
server_port=8090
num_rounds=10
num_clients=14
model=resnet18


for algorithm in fedavg splitfed_v1 splitfed_v2 dynsl; do

# start the server
srun \
    --ntasks=1 \
    --het-group=0 \
python exp/scripts/py/server.py \
    server.port=$server_port \
    general.rounds=$num_rounds \
    algorithm=$algorithm \
    partitioning=size \
    model=$model \
    strategy.min_fit_clients=$num_clients \
    wandb=true \
    experiment_name=imbalanced_sizes \
    strategy.min_evaluate_clients=0 \
    strategy.fraction_evaluate=0.0 \
    strategy.min_available_clients=$num_clients &

# wait some time for the server to start up
sleep 20

# start the client(s)
for ((client_idx=0; client_idx<num_clients; client_idx++)); do

srun \
    --ntasks=1 \
    --het-group=1 \
    --output=/dev/null \
    --error=/dev/null \
python exp/scripts/py/client.py \
    client.server_address="${server_ip}:${server_port}" \
    partitioning=size \
    model=$model \
    client.init_kwargs.cid=$client_idx \
    algorithm=$algorithm &

done
wait

done
