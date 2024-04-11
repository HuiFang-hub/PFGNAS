# -*- coding: utf-8 -*-
# @Time    : 2023/12/21 13:19
# @Function:

def generate_data_partition(init_cfg):
    if init_cfg.data.mode == 'disjoint':
        from src.FedPUB.data.generators.disjoint import generate_data
        # for n_clients in init_cfg.federate.client_num:
        config = generate_data(init_cfg)
    else:  # disable
        from src.FedPUB.data.generators.overlapping import generate_data
        comms = [2, 5]
        for n_comms in comms:
            generate_data(dataset=init_cfg.data.type, n_comms=n_comms)
    return config