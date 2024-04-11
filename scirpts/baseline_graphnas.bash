#!/bin/bash
# bash scripts/baseline_graphnas.bash

#data_types=("cora")
#client_nums=(5 10)
#actions=('a' 'b' 'c')
#for data_type in "${data_types[@]}"; do
#    for client_num in "${client_nums[@]}"; do
#      for act in "${actions[@]}"; do
#        python baseline_GraphNAS.py --cfg configs/baseline_gnns.yaml device 3 data.type $data_type federate.client_num $client_num model.actions $act federate.total_round_num 100
#done
#done
#done

#data_types=("citeseer" "pubmed")
#client_nums=(3 5 10)
#for data_type in "${data_types[@]}"; do
#    for client_num in "${client_nums[@]}"; do
#        python baseline_fl-graphnas.py --cfg configs/FLGraphNAS_cora.yaml model.type fl-graphnas device 3 data.type $data_type federate.client_num $client_num  federate.total_round_num 100
#done
#done

#data_types=("citeseer")
#client_nums=(10)
#for data_type in "${data_types[@]}"; do
#    for client_num in "${client_nums[@]}"; do
#        python baseline_fl-graphnas.py --cfg configs/FLGraphNAS_cora.yaml model.type fl-graphnas device 3 data.type $data_type federate.client_num $client_num  federate.total_round_num 100
#done
#done

#data_types=("pubmed")
#client_nums=(3)
#for data_type in "${data_types[@]}"; do
#    for client_num in "${client_nums[@]}"; do
#        declare -a splitter_args=("[{'alpha': 10}]")
#         python baseline_fl-graphnas.py --cfg configs/FLGraphNAS_cora.yaml \
#            device 2 \
#            data.splitter 'lda' \
#            data.splitter_args "${splitter_args[@]}" \
#            data.type "$data_type" \
#            federate.client_num "$client_num" \
#            train.optimizer.lr 0.005 \
#            federate.total_round_num 100 \
#            results_DIR 'results_lda' \
#            response_DIR 'exp_lda'
#
#    done
#done


data_types=("pubmed")
client_nums=(3)
for data_type in "${data_types[@]}"; do
    for client_num in "${client_nums[@]}"; do
        declare -a splitter_args=("[{'alpha': 10}]")
         python baseline_fl-graphnas.py --cfg configs/baseline_fl-graphnas_lda.yaml \
            device 2 \
            data.splitter 'lda' \
            data.splitter_args "${splitter_args[@]}" \
            data.type "$data_type" \
            federate.client_num "$client_num" \
            train.optimizer.lr 0.005 \
            federate.total_round_num 100 \
            results_DIR 'results_lda' \
            response_DIR 'exp_lda'

    done
done