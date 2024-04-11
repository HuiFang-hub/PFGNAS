# -*- coding: utf-8 -*-
# @Time    : 2023/11/3 12:06
# @Function:
from src.federatedscope.register import register_model

from src.graphnas_variants.macro_graphnas.pyg.pyg_gnn_layer import GeoLayer
import torch.nn as nn
from src.graphnas.search_space import act_map
from src.graphnas.utils.model_utils import process_action
from src.graphnas_variants.macro_graphnas.pyg.pyg_gnn import GraphNet
import torch.nn.functional as F
def get_actions(act):
    model_name_dict={'a':['gat', 'sum', 'linear', 4, 128, 'linear', 'sum', 'elu', 8, 6],
                     'b':['gcn', 'sum', 'tanh', 6, 64, 'cos', 'sum', 'tanh', 6, 3],
                     'c':['const', 'sum', 'relu6', 2, 128, 'gat', 'sum', 'linear', 2, 7]}
    actions = model_name_dict[act]
    return actions

class graphnas_module(nn.Module):
    def __init__(self,model_config,device):
        super(graphnas_module, self).__init__()
        self.multi_label = False
        self.in_feats =model_config.input_shape[-1]
        self.n_classes = model_config.num_classes
        actions = get_actions(model_config.actions)
        self.actions = process_action( actions, type = 'two',n_classes = self.n_classes)  # already modify the dimension of the last layer
        self.device = device
        self.residual = False
        self.batch_normal = True
        self.dropout = model_config.dropout
        self.build_model(self.actions, batch_normal=True, drop_out=self.dropout,
                         num_feat=self.in_feats , state_num=5)

    def build_model(self, actions, batch_normal, drop_out, num_feat, state_num):
        if self.residual:
            self.fcs = nn.ModuleList()
        if self.batch_normal:
            self.bns = nn.ModuleList()
        self.layers = nn.ModuleList()
        self.acts = []
        self.gates = nn.ModuleList()
        self.layer_nums = self.evalate_actions(actions, state_num)
        self.build_hidden_layers(actions, batch_normal, drop_out, self.layer_nums, num_feat,  state_num)


        # self.build_hidden_layers(actions, batch_normal, drop_out, self.layer_nums, num_feat, state_num)

    def evalate_actions(self, actions, state_num):
        state_length = len(actions)
        if state_length % state_num != 0:
            raise RuntimeError("Wrong Input: unmatchable input")
        layer_nums = state_length // state_num
        if self.evaluate_structure(actions, layer_nums, state_num=state_num):
            pass
        else:
            raise RuntimeError("wrong structure")
        return layer_nums

    def evaluate_structure(self, actions, layer_nums, state_num=6):
        hidden_units_list = []
        out_channels_list = []
        for i in range(layer_nums):
            head_num = actions[i * state_num + 3]
            out_channels = actions[i * state_num + 4]
            hidden_units_list.append(head_num * out_channels)
            out_channels_list.append(out_channels)

        return out_channels_list[-1] == self.n_classes

    def build_hidden_layers(self, actions, batch_normal, drop_out, layer_nums, num_feat, state_num=6):

        # build hidden layer
        for i in range(layer_nums):

            if i == 0:
                in_channels = num_feat
            else:
                in_channels = out_channels * head_num

            # extract layer information
            attention_type = actions[i * state_num + 0]
            aggregator_type = actions[i * state_num + 1]
            act = actions[i * state_num + 2]
            head_num = actions[i * state_num + 3]
            out_channels = actions[i * state_num + 4]
            concat = True
            if i == layer_nums - 1:
                concat = False
            if batch_normal:
                self.bns.append(nn.BatchNorm1d(in_channels, momentum=0.5))
            self.layers.append(
                GeoLayer(in_channels, out_channels, head_num, concat, dropout=self.dropout,
                         att_type=attention_type, agg_type=aggregator_type ))
            self.acts.append(act_map(act))
            if self.residual:
                if concat:
                    self.fcs.append(nn.Linear(in_channels, out_channels * head_num))
                else:
                    self.fcs.append(nn.Linear(in_channels, out_channels))

    def forward(self, x, edge_index_all):
        output = x
        if self.residual:
            for i, (act, layer, fc) in enumerate(zip(self.acts, self.layers, self.fcs)):
                output = F.dropout(output, p=self.dropout, training=self.training)
                if self.batch_normal:
                    output = self.bns[i](output)

                output = act(layer(output, edge_index_all) + fc(output))
        else:
            for i, (act, layer) in enumerate(zip(self.acts, self.layers)):
                output = F.dropout(output, p=self.dropout, training=self.training)
                if self.batch_normal:
                    output = self.bns[i](output)
                output = act(layer(output, edge_index_all))
        if not self.multi_label:
            output = F.log_softmax(output, dim=1)
        return output





# def get_model(model_config,device):
#     manager = graphnas_module(model_config,device)
#     return manager




def call_my_net(model_config, device):
    # Please name your gnn model with prefix 'gnn_'
    model = None
    if model_config.type.lower() == "graphnas":
        model = graphnas_module(model_config,device)
    return model


register_model("GraphNAS", call_my_net)