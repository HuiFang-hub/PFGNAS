#!/bin/bash
# bash scripts/baseline_fedpub.bash

# data_types=("cora" "citeseer" "pubmed")
# client_nums=(3 5 10)
# seeds=(1 2 3)
# for data_type in "${data_types[@]}"; do
#     for client_num in "${client_nums[@]}"; do
#       for seed in "${seeds[@]}"; do
#         python baseline_fedpub.py --cfg configs/baseline_fedpub.yaml gpu_list [2,3,4] data.type $data_type federate.client_num $client_num federate.total_round_num 100 seed $seed
# done
# done
# done

data_types=("citeseer")
client_nums=(20)
seeds=(3)
for data_type in "${data_types[@]}"; do
    for client_num in "${client_nums[@]}"; do
      for seed in "${seeds[@]}"; do
        python baseline_fedpub.py --cfg configs/baseline_fedpub.yaml gpu_list [1,2,3] data.type $data_type federate.client_num $client_num federate.total_round_num 100 seed $seed
done
done
done



