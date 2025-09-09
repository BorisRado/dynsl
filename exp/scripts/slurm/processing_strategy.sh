#!/bin/bash -l

#SBATCH --time=4:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=1GB
#SBATCH --gpus=1
#SBATCH --constraint=gpu_p100
#SBATCH --out=exp/logs/test_processing_server.log

#SBATCH hetjob
#SBATCH --ntasks=20
#SBATCH --cpus-per-task=2
#SBATCH --mem-per-cpu=1GB

srun --het-group=0 nvidia-smi

server_ip=$(srun --het-group=0 hostname)
echo "First group hosts: ---${server_ip}---"

mamba activate slower

# for now, simple test with one client and one server to check that everything works
server_port=8089
num_rounds=10
num_clients=20
algorithm=splitfed_v2
model=resnet18

for seed in 4 8 15 16 23 42; do # 4 8 15 16 23 42
for num_layers in 4; do
for batch_size in 16 128; do
for num_process_batches in 1 2 5 10 20; do

# start the server
srun \
    --ntasks=1 \
    --het-group=0 \
python exp/scripts/py/server.py \
    server.port=$server_port \
    general.rounds=$num_rounds \
    algorithm=$algorithm \
    general.seed=$seed \
    wandb=true \
    experiment_name=concatenating_batches_on_server \
    server_model_servicer.num_concat_batches=$num_process_batches \
    model=$model \
    algorithm.layer_allocator_init_args.num_client_layers=$num_layers \
    client.init_kwargs.batch_size=$batch_size \
    strategy.min_fit_clients=$num_clients \
    strategy.min_evaluate_clients=0 \
    strategy.min_available_clients=$num_clients &

# wait some time for the server to start up
sleep 20

# start the client(s)
for ((client_idx=0; client_idx<num_clients; client_idx++)); do

srun \
    --ntasks=1 \
    --het-group=1 \
    --output=exp/logs/client_${client_idx}.log \
    --open-mode=append \
python exp/scripts/py/client.py \
    client.server_address="${server_ip}:${server_port}" \
    model=$model \
    general.seed=$seed \
    client.init_kwargs.batch_size=$batch_size \
    client.init_kwargs.cid=$client_idx \
    algorithm=$algorithm &

done
wait

done
done
done
done
