# -*- coding: utf-8 -*-
# @Time    : 2023/11/7 21:25
# @Function:
from src.federatedscope.register import register_model
import torch.nn as nn
import torch
import torch.nn.functional as F
from src.FLAGNNS.models import *
import src.FLAGNNS.utils as utils

def get_code(act):
    model_name_dict={'a':[5, 5, 19, 31, 8, 51, 61, 5],
                     'b':[2, 7, 19, 0, 32, 54, 55, 1],
                     'c':[5, 7, 19, 31, 19, 31, 0, 2],
                     'd':[2, 9, 19, 0, 26, 0, 41, 5],
                     'e':[5, 9, 19, 19, 11, 38, 50, 1],
                     'f':[5, 10, 9, 11, 21, 11, 11, 2],
                     'g':[3, 10, 10, 10, 46, 22, 10, 2],
                     'h':[4, 7, 19, 17, 19, 19, 10, 2]

                     }
    codes = model_name_dict[act]
    return codes

def get_act_str(code_list):
    act_str = '-'.join(map(str, code_list))
    return act_str

def get_code_list(act_str):
    code_list =  [int(x) for x in act_str.split('-')]
    return code_list


def call_my_net(model_config, device):
    # Please name your model with prefix 'fl_'
    model = None
    nfeat, nclass = model_config.input_shape[-1], model_config.num_classes
    if model_config.type.lower() == "fl-graphnas":
        supermask = get_code_list(model_config.actions)
        model = eval('SonNet')(supermask, nfeat, nclass).to(device)
    elif model_config.type.lower() == "fl-random":
        supermask = utils.random_supermask()
        model = eval('SonNet')(supermask, nfeat, nclass).to(device)
    elif model_config.type.lower() == "fl-agnns":
        supermask = get_code(model_config.actions)
        model = eval('SuperNet')(nfeat, nclass, supermask).to(device)
    # elif model_config.type.lower() == "fl-darts":
    #     model = eval('Darts')(nfeat, nclass).to(device)
    elif model_config.type.lower() == "fl-fednas":
        model = eval('FedNas')(nfeat, nclass).to(device)

    return model


register_model("fl-graphnas", call_my_net)
register_model("fl-random", call_my_net)
register_model("fl-agnns", call_my_net)
# register_model("fl-darts", call_my_net)
# register_model("fl-fednas", call_my_net)


#
# if __name__ == '__main__':
#     device = torch.device("cuda:" + str(2))
#     nfeat, nclass = 555,888
#     supermask = [5, 0, 0, 0, 0, 0, 0, 2]
#     model = eval('SonNet')(supermask, nfeat, nclass).to(device)
#     print(model)