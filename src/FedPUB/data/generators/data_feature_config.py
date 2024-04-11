# -*- coding: utf-8 -*-
# @Time    : 2023/12/21 15:34
# @Function:
import os
import torch

def torch_load(base_dir, filename):
    fpath = os.path.join(base_dir, filename)
    return torch.load(fpath, map_location=torch.device('cpu'))

def add_data_config(config):
    dataset =config.data.type.lower()
    data_path = os.path.join(config.data.root, f'{dataset}')
    if config.data.splitter == 'lda':
        alpha = config.data.splitter_args[0]['alpha']
        global_dataset = torch_load( data_path,f'{config.data.mode}/{config.federate.client_num}_{alpha}/partition_global.pt')['client_data']
    else:
        global_dataset = torch_load( data_path,f'{config.data.mode}/{config.federate.client_num}/partition_global.pt')['client_data']

    # global_dataset = torch_load( data_path,f'{config.data.mode}/{config.federate.client_num}/partition_1.pt')['client_data']
    num_classes = len(set(global_dataset.y.tolist()))
    # dataset = [ds for ds in dataset]
    # client_num = min(len(dataset), config.federate.client_num
    #                  ) if config.federate.client_num > 0 else len(dataset)
    # config.merge_from_list(['federate.client_num', client_num])
    config.merge_from_list(['model.num_classes', num_classes])
    config.merge_from_list(['model.input_shape', tuple(global_dataset.x.shape)])
    return config

