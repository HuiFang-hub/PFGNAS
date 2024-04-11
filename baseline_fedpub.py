import os
import json
from lib.extarct_final_result import cal_avg_client_res
from src.federatedscope.core.cmd_args import parse_args, parse_client_cfg
from src.federatedscope.core.configs.config import global_cfg
from src.FedPUB.modules.multiprocs import ParentProcess
from src.FedPUB.data.generators.get_graph_data import generate_data_partition
from src.federatedscope.core.auxiliaries.logging import update_logger,get_resfile_path
import torch.multiprocessing as mp
import torch
from src.FedPUB.data.generators.data_feature_config import add_data_config
from lib.process_path import process_path
import glob
def generate_data(init_cfg):
    from src.FedPUB.data.generators.disjoint import generate_data
    modified_cfg = generate_data(init_cfg)
    init_cfg.merge_from_other_cfg(modified_cfg)
    return init_cfg
import sys
if __name__ == '__main__':
    '''
    unify_data: Common data set for all models
    only_fedup_data: original data of this method
    '''

    # data = 'unify_data'
    init_cfg = global_cfg.clone()
    args = parse_args()
    if args.cfg_file:
        init_cfg.merge_from_file(args.cfg_file)
    cfg_opt, client_cfg_opt = parse_client_cfg(args.opts)
    init_cfg.merge_from_list(cfg_opt)
    init_cfg.federate.n_workers = init_cfg.federate.client_num+1
    init_cfg = process_path(init_cfg)

    # ## generate data
    # if init_cfg.data.mode in ['disjoint']:
    #     init_cfg = generate_data(init_cfg)
    # elif init_cfg.data.mode == 'unify_data':
    #     from src.federatedscope.core.auxiliaries.data_builder import get_data
    #     data, modified_cfg = get_data(config=init_cfg.clone())

    init_cfg = add_data_config(init_cfg)
    # results dir
    # init_cfg.results_DIR = os.path.join(f'{init_cfg.results_DIR}',
    #                                     f"{init_cfg.federate.method}_{init_cfg.data.mode}_{init_cfg.data.type}_{init_cfg.federate.client_num}")


    path = init_cfg.results_DIR
    if init_cfg.data.splitter == 'lda':
        alpha = init_cfg.data.splitter_args[0]['alpha']
        init_cfg.results_DIR = os.path.join(f'{init_cfg.results_DIR}',
                                            f"{init_cfg.federate.method}_{init_cfg.data.mode}_{init_cfg.data.type}_{init_cfg.federate.client_num}_{alpha}")
    else:
        init_cfg.results_DIR = os.path.join(f'{init_cfg.results_DIR}',
                                            f"{init_cfg.federate.method}_{init_cfg.data.mode}_{init_cfg.data.type}_{init_cfg.federate.client_num}")
    init_cfg.expname = f"{init_cfg.dataloader.batch_size}" \
                       f"_{init_cfg.train.optimizer.lr}_{init_cfg.federate.total_round_num}_{init_cfg.train.local_update_steps}"
    res_dir = os.path.join(init_cfg.results_DIR, init_cfg.expname)
    avg_res_dir = os.path.join(res_dir, 'avg_res.log')
    _, init_cfg = get_resfile_path(init_cfg)

    # fl
    if init_cfg.federate.method == 'FedPUB':
        from src.FedPUB.models.fedpub.server import Server
        from src.FedPUB.models.fedpub.client import Client
    elif init_cfg.federate.method == 'fedavg':
        from src.FedPUB.models.fedavg.server import Server
        from src.FedPUB.models.fedavg.client import Client
    else:
        print('incorrect model was given: {}'.format(init_cfg.federate.method))
        os._exit(0)



    pp = ParentProcess(init_cfg, Server, Client)
    # pp = ParentProcess(args, Server, Client)
    pp.start()
    # print( os.path.join(init_cfg.results_dir, "finally_res.log"))

    dir_path = os.path.join(init_cfg.results_dir, "finally_res.log")
    clients_res = []
    for i, client_p in enumerate(glob.glob(os.path.join(init_cfg.results_dir, 'client*.txt'))):
        with open(client_p) as f:
            client_res = json.loads(f.read())
            clients_res.append(client_res['log'])
    # compute avg
    mean_values_dict = {}
    for metric in clients_res[0].keys():
        if metric in ['rnd_local_test_acc', 'rnd_local_test_roc_auc']:
            new_metric = 'acc' if metric == 'rnd_local_test_acc' else 'roc_auc'
            values = [item[metric] for item in clients_res]
            mean_values_dict[new_metric] = [sum(v) / len(v) for v in zip(*values)]
    with open(dir_path, 'a') as f:
        json.dump(mean_values_dict, f)
        f.write('\n')
    # compte res
    # avg_ONE_ROUND_res = cal_avg_client_res(res_dir)
    # print(avg_ONE_ROUND_res)
    # with open(os.path.join(init_cfg.results_dir,'finally_res.log') , 'a') as f:
    #     json.dump(avg_ONE_ROUND_res, f)
    # f.write('\n')

    # count = get_resfile_path(init_cfg)
    sys.exit()
