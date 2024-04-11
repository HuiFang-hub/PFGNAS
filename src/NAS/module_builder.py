# -*- coding: utf-8 -*-
# @Time    : 2023/10/17 10:00
# @Function:
from src.federatedscope.gfl.model import get_model
import scipy.sparse as sp

def get_strcut_dict():
    struct_dict = {'0000':
                       [[0, 1, 1, 1, 1, 0],
                        [0, 0, 0, 0, 0, 1],
                        [0, 0, 0, 0, 0, 1],
                        [0, 0, 0, 0, 0, 1],
                        [0, 0, 0, 0, 0, 1],
                        [0, 0, 0, 0, 0, 0]],

                   '0001':
                       [[0, 1, 1, 1, 0, 0],
                        [0, 0, 0, 0, 1, 0],
                        [0, 0, 0, 0, 0, 1],
                        [0, 0, 0, 0, 0, 1],
                        [0, 0, 0, 0, 0, 1],
                        [0, 0, 0, 0, 0, 0]],

                   '0011':
                       [[0, 1, 1, 0, 0, 0],
                        [0, 0, 0, 1, 1, 0],
                        [0, 0, 0, 0, 0, 1],
                        [0, 0, 0, 0, 0, 1],
                        [0, 0, 0, 0, 0, 1],
                        [0, 0, 0, 0, 0, 0]],

                   '0012':
                       [[0, 1, 1, 0, 0, 0],
                        [0, 0, 0, 1, 0, 0],
                        [0, 0, 0, 0, 1, 0],
                        [0, 0, 0, 0, 0, 1],
                        [0, 0, 0, 0, 0, 1],
                        [0, 0, 0, 0, 0, 0]],

                   '0013':
                       [[0, 1, 1, 0, 0, 0],
                        [0, 0, 0, 1, 0, 0],
                        [0, 0, 0, 0, 0, 1],
                        [0, 0, 0, 0, 1, 0],
                        [0, 0, 0, 0, 0, 1],
                        [0, 0, 0, 0, 0, 0]],

                   '0111':
                       [[0, 1, 0, 0, 0, 0],
                        [0, 0, 1, 1, 1, 0],
                        [0, 0, 0, 0, 0, 1],
                        [0, 0, 0, 0, 0, 1],
                        [0, 0, 0, 0, 0, 1],
                        [0, 0, 0, 0, 0, 0]],

                   '0112':
                       [[0, 1, 0, 0, 0, 0],
                        [0, 0, 1, 1, 0, 0],
                        [0, 0, 0, 0, 1, 0],
                        [0, 0, 0, 0, 0, 1],
                        [0, 0, 0, 0, 0, 1],
                        [0, 0, 0, 0, 0, 0]],

                   '0122':
                       [[0, 1, 0, 0, 0, 0],
                        [0, 0, 1, 0, 0, 0],
                        [0, 0, 0, 1, 1, 0],
                        [0, 0, 0, 0, 0, 1],
                        [0, 0, 0, 0, 0, 1],
                        [0, 0, 0, 0, 0, 0]],

                   '0123':
                       [[0, 1, 0, 0, 0, 0],
                        [0, 0, 1, 0, 0, 0],
                        [0, 0, 0, 1, 0, 0],
                        [0, 0, 0, 0, 1, 0],
                        [0, 0, 0, 0, 0, 1],
                        [0, 0, 0, 0, 0, 0]]

                   }
    return struct_dict

def get_adj_matrx(struct):
    struct_dict = get_strcut_dict()
    adjacency_model = struct_dict.get(struct)
    return adjacency_model

def get_model_name(name_str):
    model_name_dict={'a':'gcn','b':'sage', 'c':'gpr', 'd':'gat','e': 'gin', 'f':'fc',
                     'g':'sgc', 'h':'arma', 'i':'appnp'}
    model_list = []
    for i in name_str:
        model_list.append(model_name_dict[i])
    return model_list

def get_model_name_short(model_list):
    model_name_dict = {'gcn':'a', 'sage':'b', 'gpr': 'c', 'gat': 'd', 'gin': 'e', 'fc': 'f',
                       'sgc':'g', 'arma':'h', 'appnp':'i'}
    model_str = ''
    for i in model_list:
        model_str += model_name_dict[i.lower()]
    return model_str
# def select_model(model_config,device):
#     models = model_config.model_list
#     input_shape, hidden_shape = model_config.input_shape[-1],model_config.hidden
#     if len(models) != 4:
#         raise ValueError("There are not 4 operations!")
#     model_list = []
#     for i in range(len(models)):
#         if models[i].lower() in ['gcn', 'sage', 'gpr', 'gat', 'gin', 'mpnn']:
#             if i == 0:
#                 model = get_model(models[i], model_config, input_shape, hidden_shape).to(device)
#             # elif i == len(models)-1:
#             #     model = get_gnn(models[i], hidden_shape,out_shape)
#             else:
#                 model = get_model(models[i], model_config, hidden_shape, hidden_shape).to(device)
#             model_list.append(model)
#
#     return model_list


def select_model(model_config,csr_matrix,device):
    models = get_model_name(model_config.operations)
    input_shape, hidden_shape,output_shape = model_config.input_shape[-1],model_config.hidden,model_config.num_classes
    if len(models) != 4:
        raise ValueError("There are not 4 operations!")
    model_dict = {}
    _,target_indices = csr_matrix[0, :].nonzero()  # first layer
    first_model= [x - 1 for x in target_indices]
    source_indices, _ = csr_matrix[:, len(models) + 1].nonzero() # last layer
    last_model = [x - 1 for x in source_indices]

    for i in range(len(models)):
        if i in first_model:
            input_s = input_shape
        else:
            input_s = hidden_shape
        if i in last_model:
            output_s = output_shape
        else:
            output_s = hidden_shape
        model = get_model(models[i], model_config, input_s, output_s).to(device)
        # parma = list(model.parameters())
        # print(f"models[i]: ",parma)
        model_dict[i] = model

    return model_dict
    #
    # for start in csr_matrix.indices[target_indices]:
    #     model = get_model(models[start], model_config, input_shape, hidden_shape).to(device)
    #     model_dict[start] = model
    #
    #
    # for end in source_indices:
    #     model = get_model(models[end], model_config, hidden_shape, output_shape).to(device)
    #     model_dict[end] = model






    # for i in range(len(models)):
    #     if models[i].lower() in ['gcn', 'sage', 'gpr', 'gat', 'gin', 'mpnn']:
    #         if i == 0:
    #             model = get_model(models[i], model_config, input_shape, hidden_shape).to(device)
    #         # elif i == len(models)-1:
    #         #     model = get_gnn(models[i], hidden_shape,out_shape)
    #         else:
    #             model = get_model(models[i], model_config, hidden_shape, hidden_shape).to(device)
    #         model_list.append(model)





