import argparse
import sys
from src.federatedscope.core.configs.config import global_cfg
import sys
import os
import argparse

current_file_name = __file__
current_dir=os.path.dirname(os.path.dirname(os.path.abspath(current_file_name)))
if current_dir not in sys.path:
    sys.path.append(current_dir)
# def parse_args(args=None):
#     parser = argparse.ArgumentParser(description='src.federatedscope',
#                                      add_help=False)
#     parser.add_argument('--cfg',
#                         dest='cfg_file',
#                         # default='gfl/baseline/fedavg_gnn_node_fullbatch_citation.yaml',
#                         help='Config file path',
#                         required=False,
#                         type=str)
#     parser.add_argument('--client_cfg',
#                         dest='client_cfg_file',
#                         help='Config file path for clients',
#                         required=False,
#                         default=None,
#                         type=str)
#     parser.add_argument('--dataset',
#                         type=str,
#                         default='Cora',
#                         help='Dataset used in the experiment')
#     parser.add_argument('--feat_dim',
#                         type=int,
#                         default=1433,
#                         help='number of features dimension. Defaults to 1000.')
#     parser.add_argument('--num_nodes',
#                         type=int,
#                         default=2708,
#                         help='number of nodes. Defaults to 2708.')
#     parser.add_argument('--data_path',
#                         type=str,
#                         default='src/src.dgld/data/',
#                         help='data path')
#     parser.add_argument('--device',
#                         type=str,
#                         default='0',
#                         help='ID(s) of gpu used by cuda')
#     parser.add_argument('--seed',
#                         type=int,
#                         default=4096,
#                         help='Random seed. Defaults to 4096.')
#     parser.add_argument('--save_path',
#                         type=str,
#                         help='save path of the result')
#     parser.add_argument('--exp_name',
#                         type=str,
#                         help='exp_name experiment identification')
#     parser.add_argument('--runs',
#                         type=int,
#                         default=1,
#                         help='The number of runs of task with same parmeter,If the number of runs is not 1, \
#                                 we will randomly generate different seeds to calculate the variance')
#     parser.add_argument(
#         '--help',
#         nargs="?",
#         const="all",
#         default="",
#     )
#     parser.add_argument('opts',
#                         help='See src.federatedscope/core/configs for all options',
#                         default=None,
#                         nargs=argparse.REMAINDER)
#     # get dataset
#     arg_list = sys.argv[1:]
#     if '--dataset' in arg_list:
#         idx = arg_list.index('--dataset') + 1
#         dataset = arg_list[idx]
#     elif any(map(lambda x: x.startswith('--dataset='), arg_list)):
#         dataset = [x.split("=")[-1] for x in arg_list if x.startswith('--dataset=')]
#         dataset = dataset[0]
#     else:
#         dataset = parser.get_default('dataset')
#
#     # set default feat_dim and num_nodes
#     if dataset in IN_FEATURE_MAP.keys():
#         parser.set_defaults(feat_dim=IN_FEATURE_MAP[dataset], num_nodes=NUM_NODES_MAP[dataset])
#
#     subparsers = parser.add_subparsers(dest="model", help='sub-command help')
#
#     # set sub args
#     for _model, set_arg_func in models_set_args_map.items():
#         sub_parser = subparsers.add_parser(
#             _model, help=f"Run anomaly detection on {_model}")
#         set_arg_func(sub_parser)
#
#         # set best args
#         fp = f'src/src.dgld/config/{_model}.json'
#         if os.path.exists(fp):
#             best_config = loadargs_from_json(fp)
#             sub_parser.set_defaults(**best_config.get(dataset, {}))
#
#     args_dict, args = models_get_args_map[args.model](args)
#     parse_res = parser.parse_args(args)
#     init_cfg = global_cfg.clone()
#     # when users type only "fs_main.py" or "fs_main.py help"
#     # print(parse_res.cfg)
#     if (len(sys.argv) == 1 and parse_res.cfg==None) or parse_res.help == "all":
#         parser.print_help()
#         init_cfg.print_help()
#         sys.exit(1)
#     elif hasattr(parse_res, "help") and isinstance(
#             parse_res.help, str) and parse_res.help != "":
#         init_cfg.print_help(parse_res.help)
#         sys.exit(1)
#     elif hasattr(parse_res, "help") and isinstance(
#             parse_res.help, list) and len(parse_res.help) != 0:
#         for query in parse_res.help:
#             init_cfg.print_help(query)
#         sys.exit(1)
#
#     return parse_res,args_dict
#

def parse_args(args=None):
    parser = argparse.ArgumentParser(description='federatedscope',
                                     add_help=False)
    parser.add_argument('--cfg',
                        dest='cfg_file',
                        # default='gfl/baseline/fedavg_gnn_node_fullbatch_citation.yaml',
                        help='Config file path',
                        required=False,
                        type=str)
    parser.add_argument('--client_cfg',
                        dest='client_cfg_file',
                        help='Config file path for clients',
                        required=False,
                        default=None,
                        type=str)
    parser.add_argument(
        '--help',
        nargs="?",
        const="all",
        default="",
    )
    parser.add_argument('opts',
                        help='See federatedscope/core/configs for all options',
                        default=None,
                        nargs=argparse.REMAINDER)
    parse_res = parser.parse_args(args)
    init_cfg = global_cfg.clone()
    # when users type only "main.py" or "main.py help"
    # print(parse_res.cfg)
    if (len(sys.argv) == 1 and parse_res.cfg==None) or parse_res.help == "all":
        parser.print_help()
        init_cfg.print_help()
        sys.exit(1)
    elif hasattr(parse_res, "help") and isinstance(
            parse_res.help, str) and parse_res.help != "":
        init_cfg.print_help(parse_res.help)
        sys.exit(1)
    elif hasattr(parse_res, "help") and isinstance(
            parse_res.help, list) and len(parse_res.help) != 0:
        for query in parse_res.help:
            init_cfg.print_help(query)
        sys.exit(1)

    return parse_res




def parse_client_cfg(arg_opts):
    """
    Arguments:
        arg_opts: list pairs of arg.opts
    """
    client_cfg_opts = []
    i = 0
    while i < len(arg_opts):
        if arg_opts[i].startswith('client'):
            client_cfg_opts.append(arg_opts.pop(i))
            client_cfg_opts.append(arg_opts.pop(i))
        else:
            i += 1
    return arg_opts, client_cfg_opts
