from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

from src.federatedscope.gfl.model.gcn import GCN_Net
from src.federatedscope.gfl.model.sage import SAGE_Net
from src.federatedscope.gfl.model.gat import GAT_Net
from src.federatedscope.gfl.model.gin import GIN_Net
from src.federatedscope.gfl.model.gpr import GPR_Net
from src.federatedscope.gfl.model.link_level import GNN_Net_Link
from src.federatedscope.gfl.model.graph_level import GNN_Net_Graph
from src.federatedscope.gfl.model.mpnn import MPNNs2s
from src.federatedscope.gfl.model.sgc import SGC
from src.federatedscope.gfl.model.arma import ARMA
from src.federatedscope.gfl.model.appnp import APPNP
from src.federatedscope.gfl.model.linear import LinearConv
from src.federatedscope.gfl.model.identity import Identity,ZeroConv
import torch.nn as nn

def get_gnn(model_config, input_shape,out_shape):

    x_shape, num_label, num_edge_features = input_shape
    if not num_label:
        num_label = 0
    if model_config.task.startswith('node'):
        if model_config.type == 'gcn':
            # assume `data` is a dict where key is the client index,
            # and value is a PyG object
            # model = GCN_Net(x_shape[-1],
            #                 out_channels=out_shape,
            #                 hidden=model_config.hidden,
            #                 max_depth=model_config.layer,
            #                 dropout=model_config.dropout)
            model = GCN_Net(x_shape[-1],
                            out_channels=out_shape,
                            hidden=model_config.hidden,
                            max_depth=model_config.layer,
                            dropout=model_config.dropout)
        elif model_config.type == 'sage':
            model = SAGE_Net(x_shape[-1],
                             out_channels=out_shape,
                             hidden=model_config.hidden,
                             max_depth=model_config.layer,
                             dropout=model_config.dropout)
        elif model_config.type == 'gat':
            model = GAT_Net(x_shape[-1],
                            out_channels=out_shape,
                            hidden=model_config.hidden,
                            max_depth=model_config.layer,
                            dropout=model_config.dropout)
        elif model_config.type == 'gin':
            model = GIN_Net(x_shape[-1],
                            out_channels=out_shape,
                            hidden=model_config.hidden,
                            max_depth=model_config.layer,
                            dropout=model_config.dropout)
        elif model_config.type == 'gpr':
            model = GPR_Net(x_shape[-1],
                            out_channels=out_shape,
                            hidden=model_config.hidden,
                            K=model_config.layer,
                            dropout=model_config.dropout)
        else:
            raise ValueError('not recognized gnn model {}'.format(
                model_config.type))

    elif model_config.task.startswith('link'):
        model = GNN_Net_Link(x_shape[-1],
                             out_channels=out_shape,
                             hidden=model_config.hidden,
                             max_depth=model_config.layer,
                             dropout=model_config.dropout,
                             gnn=model_config.type)
    elif model_config.task.startswith('graph'):
        if model_config.type == 'mpnn':
            model = MPNNs2s(in_channels=x_shape[-1],
                            out_channels =out_shape,
                            num_nn=num_edge_features,
                            hidden=model_config.hidden)
        else:
            model = GNN_Net_Graph(x_shape[-1],
                                  out_channels=max(out_shape, num_label),
                                  hidden=model_config.hidden,
                                  max_depth=model_config.layer,
                                  dropout=model_config.dropout,
                                  gnn=model_config.type,
                                  pooling=model_config.graph_pooling)
    else:
        raise ValueError('not recognized data task {}'.format(
            model_config.task))
    return model


def get_model(model_name, model_config,input_shape, out_shape):
    # input_shape,out_shape = model_config.input_shape,model_config.hidden
    # ['gcn', 'sage', 'gpr', 'gat', 'gin', 'fc', 'sgc', 'arma', 'appnp']

    if model_name == 'gcn':
        # assume `data` is a dict where key is the client index,
        # and value is a PyG object
        # model = GCN_Net(input_shape,
        #                 out_channels=out_shape,
        #                 hidden=model_config.hidden,
        #                 max_depth=model_config.layer,
        #                 dropout=model_config.dropout)
        model = GCN_Net(input_shape,
                        out_channels=out_shape,
                        hidden=model_config.hidden,
                        )
    elif model_name == 'sage':
        model = SAGE_Net(input_shape,
                         out_channels=out_shape,
                         hidden=model_config.hidden,
                         max_depth=model_config.layer,
                         dropout=model_config.dropout)
    elif model_name == 'gpr':
        model = GPR_Net(input_shape,
                        out_channels=out_shape,
                        hidden=model_config.hidden,
                        K=model_config.layer,
                        dropout=model_config.dropout)
    elif model_name == 'gat':
        # model = GAT_Net(input_shape,
        #                 out_channels=out_shape,
        #                 hidden=model_config.hidden,
        #                 max_depth=model_config.layer,
        #                 dropout=model_config.dropout)
        model = GAT_Net(input_shape,
                        out_channels=out_shape,hidden=model_config.hidden,
                        )
    elif model_name == 'gin':
        model = GIN_Net(input_shape,
                        out_channels=out_shape,
                        hidden=model_config.hidden,
                        max_depth=model_config.layer,
                        dropout=model_config.dropout)

    elif model_name =='fc':
        # model = nn.Linear(input_shape,out_shape)
        model = LinearConv(input_shape,out_shape, bias=True)
    elif model_name == 'sgc':
        model = SGC(input_shape,out_shape)
    elif model_name == 'arma':
        model = ARMA(input_shape,out_shape,hidden=model_config.hidden)
    elif model_name == 'appnp':
        model = APPNP(input_shape,out_shape,hidden=model_config.hidden)
    elif model_name == 'identity':
        model = Identity()
    elif model_name == "zero":
        model = ZeroConv()

    else:
        raise ValueError('not recognized gnn model {}'.format(
            model_name))


    return model
