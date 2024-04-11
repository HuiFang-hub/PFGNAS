#!/bin/bash
# bash scripts/baseline_fl-darts.bash

#data_types=("cora")
#client_nums=(3 5 10)
#for data_type in "${data_types[@]}"; do
#    for client_num in "${client_nums[@]}"; do
#        python baseline_fl-darts.py --cfg configs/FLDarts_cora.yaml device 2 data.type $data_type federate.client_num $client_num federate.total_round_num 100
#done
#done

# data_types=("citeseer" "pubmed")
# client_nums=(3 5 10)
# for data_type in "${data_types[@]}"; do
#     for client_num in "${client_nums[@]}"; do
#         python baseline_fl-darts.py --cfg configs/FLDarts_cora.yaml device 1 data.type $data_type federate.client_num $client_num federate.total_round_num 100
# done
# done

data_types=("cora")
client_nums=(20)
for data_type in "${data_types[@]}"; do
    for client_num in "${client_nums[@]}"; do
        python baseline_fl-darts.py --cfg configs/baseline_fl-darts.yaml device 1 \
        data.type $data_type \
        federate.client_num $client_num \
        federate.total_round_num 100
done
done



