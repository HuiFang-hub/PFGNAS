# -*- coding: utf-8 -*-
# @Time    : 2023/10/17 9:58
# @Function:
from src.federatedscope.register import register_model
from torch.nn import ModuleList
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv
import torch
import torch.nn.functional as F
import scipy.sparse as sp
from src.NAS.module_builder import select_model, get_adj_matrx
import torch.nn as nn

class EmptyModule(nn.Module):
    def __init__(self):
        super(EmptyModule, self).__init__()

    def forward(self, x):
        # 这里可以留空或返回空值
        return x

class dynamic_module(nn.Module):
    def __init__(self,model_config,device):
        super(dynamic_module,self).__init__()
        self.device = device
        # self.model_list =  nn.ModuleList()
        self.csr_matrix = sp.csr_matrix(get_adj_matrx(model_config.struct))
        self.num_model = len(model_config.operations)
        model_dict = select_model(model_config,self.csr_matrix,self.device)
        self.model_list = nn.ModuleList(list(model_dict.values()))
        self.dropout = model_config.dropout
        # print("test")
        # self.output_layer = EmptyModule().to(self.device) #nn.Linear(model_config.hidden,model_config.num_classes).to(self.device) #
        # self.model_list[len(model_config.model_list)] = self.output_layer


    def forward(self, x, edge_index):
        # 转化为CSR格式
        # csr_matrix = sp.csr_matrix(self.adjacency_model)
        fea_dict = {0:x.unsqueeze(0)}
        for src, tar in zip(self.csr_matrix.nonzero()[0], self.csr_matrix.nonzero()[1]):
            input_fea = torch.mean(fea_dict[src],dim=0)
            if tar-1 == self.num_model:  # isinstance(self.model_list[tar-1], EmptyModule):
                output = input_fea
            elif isinstance(self.model_list[tar-1], nn.Linear):
                output = self.model_list[tar - 1](input_fea).unsqueeze(0)
            else:
                output = self.model_list[tar-1](input_fea,edge_index).unsqueeze(0)
            if tar not in fea_dict:
                fea_dict[tar]=output
            else:
                fea_dict[tar] = torch.cat((fea_dict[tar],output), dim=0)
        # last_key, last_value = fea_dict.popitem()
        last_value = fea_dict[len(self.model_list)]
        fea = torch.mean(last_value,dim=0)
        return  F.relu(F.dropout(fea, p=self.dropout, training=self.training))

def call_my_net(model_config, device):
    # Please name your gnn model with prefix 'gnn_'
    model = None
    if model_config.type.lower() == "nas":
        model = dynamic_module(model_config,device)
    # elif model_config.type.lower() == "cola":
    #     model =  colabuilder(model_config,input_shape,device)
    #     # model = CoLA()
    # elif model_config.type.lower() == "anemone":
    #     model = anemonebuilder(model_config, input_shape, device)
    return model

register_model("nas", call_my_net)
# if __name__ == '__main__':
#
#     x = torch.tensor([[1.0, 2.0, 3.0, 4.0, 5.0],
#                       [2.0, 3.0, 4.0, 5.0, 6.0],
#                       [3.0, 4.0, 5.0, 6.0, 7.0],
#                       [4.0, 5.0, 6.0, 7.0, 8.0]], dtype=torch.float)
#
#     edge_index = torch.tensor([[0, 1, 2, 3], [1, 2, 3, 0]], dtype=torch.long)
#     y = torch.tensor([0, 1, 0, 1], dtype=torch.long)
#
#     # data = Data(x=x, edge_index=edge_index, y=y)
#     logit = dynamic_module(model_config,input_shape,device)


