#!/bin/bash
# bash scripts/baseline_fl-agnns.bash

#data_types=("cora")
#client_nums=(3)
#actions=('a'  'c'  'e' 'f' 'g' 'h')
#for data_type in "${data_types[@]}"; do
#    for client_num in "${client_nums[@]}"; do
#       for act in "${actions[@]}"; do
#          python baseline_fl-agnns.py --cfg configs/FLAgnns_cora.yaml model.type fl-agnns device 0 model.actions $act data.type $data_type federate.client_num $client_num federate.total_round_num 100
#done
#done
#done

# data_types=("citeseer" "pubmed")
# client_nums=(3 5 10)
# actions=('a'  'c'  'e' 'f' 'g' 'h')
# for data_type in "${data_types[@]}"; do
#     for client_num in "${client_nums[@]}"; do
#        for act in "${actions[@]}"; do
#           python baseline_fl-agnns.py --cfg configs/FLAgnns_cora.yaml model.type fl-agnns device 0 model.actions $act data.type $data_type federate.client_num $client_num federate.total_round_num 100
# done
# done
# done

data_types=("citeseer")
client_nums=(20)
actions=('a'  'c'  'e' 'f' 'g' 'h')
for data_type in "${data_types[@]}"; do
    for client_num in "${client_nums[@]}"; do
       for act in "${actions[@]}"; do
          python baseline_fl-agnns.py --cfg configs/FLAgnns_cora.yaml model.type fl-agnns \
          device 3 \
          model.actions $act \
          data.type $data_type \
          federate.client_num $client_num \
          federate.total_round_num 100
done
done
done