data_types=("citeseer")
client_nums=(20)

for data_type in "${data_types[@]}"; do
    for client_num in "${client_nums[@]}"; do
        python main_pfgnas.py --cfg configs/main_pfgnas.yaml data.type $data_type \
            federate.client_num $client_num \
            federate.total_round_num 100 \
            device 5
done
done