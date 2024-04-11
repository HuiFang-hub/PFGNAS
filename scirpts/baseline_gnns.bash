#!/bin/bash
# bash scripts/baseline_gnns.bash

data_types=("citeseer" "pubmed")
client_nums=(5 10)
types=('gcn' 'sage' 'gat' 'gin' 'gpr' 'fc' 'sgc' 'arma' 'appnp')
for data_type in "${data_types[@]}"; do
   for client_num in "${client_nums[@]}"; do
     for type in "${types[@]}"; do
       python baseline_GNN.py --cfg configs/baseline_gnns.yaml device 2 data.type $data_type federate.client_num $client_num model.type $type federate.total_round_num 100
done
done
done

# data_types=("pubmed")
# client_nums=(3)
# types=('gcn')
# for data_type in "${data_types[@]}"; do
#     for client_num in "${client_nums[@]}"; do
#       for type in "${types[@]}"; do
#         declare -a splitter_args=("[{'alpha': 0.2}]")
#         python baseline_GNN.py --cfg configs/baseline_gnns.yaml\
#             device 5 \
#             data.splitter_args "${splitter_args[@]}" \
#             data.type "$data_type" \
#             data.splitter 'lda'\
#             model.type $type \
#             federate.method 'FedAvg' \
#             federate.client_num "$client_num" \
#             train.optimizer.lr 0.005 \
#             federate.total_round_num 100\
#             results_DIR 'results_lda' \
#             response_DIR 'exp_lda'
#     done
# done
# done


# data_types=("cora")
# client_nums=(20)
# types=('gcn' 'sage' 'gat' 'gin' 'gpr' 'fc' 'sgc' 'arma' 'appnp')
# for data_type in "${data_types[@]}"; do
#     for client_num in "${client_nums[@]}"; do
#       for type in "${types[@]}"; do
#         python baseline_GNN.py --cfg configs/baseline_gnns.yaml\
#             device 5 \
#             data.splitter 'louvain'\
#             model.type $type \
#             federate.method 'FedAvg' \
#             federate.client_num "$client_num" \
#             train.optimizer.lr 0.005 \
#             federate.total_round_num 100\
#             data.type "$data_type" \
#             results_DIR 'results' \
#             response_DIR 'exp'
#     done
# done
# done

data_types=("citeseer")
client_nums=(20)
types=('gcn')
for data_type in "${data_types[@]}"; do
    for client_num in "${client_nums[@]}"; do
      for type in "${types[@]}"; do
        python baseline_GNN.py --cfg configs/baseline_gnns.yaml\
            device 3 \
            data.splitter 'louvain'\
            model.type $type \
            federate.method 'FedAvg' \
            federate.client_num "$client_num" \
            train.optimizer.lr 0.005 \
            federate.total_round_num 100\
            data.type "$data_type" \
            results_DIR 'results' \
            response_DIR 'exp'
    done
done
done


