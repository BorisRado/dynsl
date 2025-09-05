#!/bin/bash -l

#SBATCH --time=4:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=1GB
#SBATCH --gpus=1
#SBATCH --out=exp/logs/test1.log

#SBATCH hetjob
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=1GB

server_ip=$(srun --het-group=0 hostname)
echo "First group hosts: ---${server_ip}---"

mamba activate slower

# for now, simple test with one client and one server to check that everything works
server_port=8089
algorithm=fedavg
num_rounds=20

for algorithm in ares fedavg splitfed_v1 splitfed_v2; do

# start the server
srun \
    --ntasks=1 \
    --het-group=0 \
python exp/scripts/py/server.py \
    server.port=$server_port \
    general.rounds=$num_rounds \
    algorithm=$algorithm &

# wait some time for the server to start up
sleep 20

# start the client(s)
srun \
    --ntasks=1 \
    --het-group=1 \
python exp/scripts/py/client.py \
    client.server_address="${server_ip}:${server_port}" \
    client.init_kwargs.client_id=0 \
    algorithm=$algorithm &

wait

done
