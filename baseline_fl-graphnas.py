# -*- coding: utf-8 -*-
# @Time    : 2023/10/26 15:04
# @Function:
import os
import sys
import logging
from lib.extarct_final_result import extarct_res, cal_path_results
import json
from src.FLAGNNS.models import *
DEV_MODE = False  # simplify the src.federatedscope re-setup everytime we change

if DEV_MODE:
    file_dir = os.path.join(os.path.dirname(__file__), '..')
    sys.path.append(file_dir)
import time
from src.federatedscope.core.cmd_args import parse_args, parse_client_cfg
from src.federatedscope.core.auxiliaries.data_builder import get_data
from src.federatedscope.core.auxiliaries.utils import setup_seed
from src.federatedscope.core.auxiliaries.logging import update_logger
from src.federatedscope.core.auxiliaries.worker_builder import get_client_cls, \
    get_server_cls
from src.federatedscope.core.configs.config import global_cfg, CfgNode
from src.federatedscope.core.auxiliaries.runner_builder import get_runner
import src.graphnas.trainer as trainer
root_logger = logging.getLogger("src.federatedscope")
from lib.process_path import process_path
# def process_path(init_cfg):
#     #  result dir
#     if init_cfg.response_DIR == "":
#         init_cfg.response_dir = os.path.join(init_cfg.response_DIR,
#                                              f"{init_cfg.federate.method}_{init_cfg.model.type}_{init_cfg.data.type}_{init_cfg.federate.client_num}")
#     else:
#         init_cfg.response_dir = os.path.join("exp",
#                                              f"{init_cfg.federate.method}_{init_cfg.model.type}_{init_cfg.data.type}_{init_cfg.federate.client_num}")
#     if not os.path.exists(init_cfg.response_dir):
#         os.makedirs(init_cfg.response_dir)

    # message_folder = os.path.join(init_cfg.response_dir, 'message')
    # response_folder = os.path.join(init_cfg.response_dir, 'response')
    # if not os.path.exists(message_folder):
    #     os.makedirs(message_folder)
    # if not os.path.exists(response_folder):
    #     os.makedirs(response_folder)
    # return init_cfg

if __name__ == '__main__':
    # init config
    init_cfg = global_cfg.clone()
    args = parse_args()
    if args.cfg_file:
        init_cfg.merge_from_file(args.cfg_file)
    cfg_opt, client_cfg_opt = parse_client_cfg(args.opts)
    init_cfg.merge_from_list(cfg_opt)

    init_cfg = process_path(init_cfg)

    if args.client_cfg_file:
        client_cfgs = CfgNode.load_cfg(open(args.client_cfg_file, 'r'))
        # client_cfgs.set_new_allowed(True)
        client_cfgs.merge_from_list(client_cfg_opt)
    else:
        client_cfgs = None

    # data config
    data, modified_cfg = get_data(config=init_cfg.clone(),
                                  client_cfgs=client_cfgs)
    init_cfg.merge_from_other_cfg(modified_cfg)


    iterations = 15

    model = GraphNas(init_cfg.model.hidden,init_cfg.device).to(init_cfg.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=init_cfg.graphnas.controller.controller_lr)
    model.train()
    all_res = {}
    all_res_dir = os.path.join(init_cfg.response_dir, 'res.log')
    with open(all_res_dir, 'a') as f:   # 'a': append mode
        json.dump('Begin a new experiment!', f)
        f.write('\n')
    path = init_cfg.results_DIR
    for iteration in range(iterations):
        # generate code
        begin = time.time()
        dummy_code = model.generate_code()
        supermask = model.parse_code(dummy_code)
        init_cfg.model.actions = '-'.join(map(str, supermask))
        print(init_cfg.model.actions)
        response_path = os.path.join(init_cfg.response_dir, 'response.log')  # store a operations combination
        with open(response_path, 'a') as f:
            json.dump(init_cfg.model.actions, f)
            f.write('\n')
            # if iteration == len(model_lists) - 1:  # Add another blank line
            #     f.write('\n')

        # results dir

        # init_cfg.results_DIR = os.path.join('results',
        #                                     f"{init_cfg.federate.method}_{init_cfg.model.type}_{init_cfg.data.type}_{init_cfg.federate.client_num}")


        if init_cfg.data.splitter == 'lda':
            alpha = init_cfg.data.splitter_args[0]['alpha']
            init_cfg.results_DIR = os.path.join(f'{path}',
                                                f"{init_cfg.federate.method}_{init_cfg.model.type}_{init_cfg.data.type}_{init_cfg.federate.client_num}_{alpha}")
        else:
            init_cfg.results_DIR = os.path.join(f'{path}',
                                                f"{init_cfg.federate.method}_{init_cfg.model.type}_{init_cfg.data.type}_{init_cfg.federate.client_num}")

        init_cfg.expname = f"{init_cfg.model.actions}_{init_cfg.dataloader.batch_size}" \
                           f"_{init_cfg.train.optimizer.lr}_{init_cfg.federate.total_round_num}_{init_cfg.train.local_update_steps}"
        res_dir = os.path.join(init_cfg.results_DIR, init_cfg.expname)
        avg_res_dir = os.path.join(res_dir, 'avg_res.log')
        count = update_logger(init_cfg, clear_before_add=True)
        # get results
        while count < 3:
            init_cfg.seed = count
            setup_seed(init_cfg.seed)
            runner = get_runner(data=data,
                                server_class=get_server_cls(init_cfg),
                                client_class=get_client_cls(init_cfg),
                                config=init_cfg.clone(),
                                client_configs=client_cfgs)
            _ = runner.run()
            root_logger.info("Done!")
            count = update_logger(init_cfg, clear_before_add=True)
        #store results
        avg_results_str,avg_results_num = cal_path_results(os.path.join(init_cfg.results_DIR, init_cfg.expname))
        with open(avg_res_dir, 'a') as f:
            json.dump(avg_results_str, f)
            f.write('\n')
        avg_acc = avg_results_num['acc']
        _time = time.time() - begin
        print(f"time long:{_time}\n")
        avg_results_str['time'] = _time
        with open(all_res_dir, 'a') as f:
            f.write('\n')
            json.dump(f'{iteration}:{avg_results_str}', f)
            f.write('\n')
            # backward
        loss = model.get_loss(dummy_code, supermask, avg_acc)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    # for iteration in range(1):
    #     # generate code
    #     begin = time.time()
    #     # dummy_code = model.generate_code()
    #     # supermask =
    #     init_cfg.model.actions = '3-9-2-9-3-5-1-4'
    #     print(init_cfg.model.actions)
    #     response_path = os.path.join(init_cfg.response_dir, 'response.log')  # store a operations combination
    #     with open(response_path, 'a') as f:
    #         json.dump(init_cfg.model.actions, f)
    #         f.write('\n')
    #         # if iteration == len(model_lists) - 1:  # Add another blank line
    #         #     f.write('\n')
    #
    #     # results dir
    #
    #     # init_cfg.results_DIR = os.path.join('results',
    #     #                                     f"{init_cfg.federate.method}_{init_cfg.model.type}_{init_cfg.data.type}_{init_cfg.federate.client_num}")
    #
    #
    #     if init_cfg.data.splitter == 'lda':
    #         alpha = init_cfg.data.splitter_args[0]['alpha']
    #         init_cfg.results_DIR = os.path.join(f'{path}',
    #                                             f"{init_cfg.federate.method}_{init_cfg.model.type}_{init_cfg.data.type}_{init_cfg.federate.client_num}_{alpha}")
    #     else:
    #         init_cfg.results_DIR = os.path.join(f'{path}',
    #                                             f"{init_cfg.federate.method}_{init_cfg.model.type}_{init_cfg.data.type}_{init_cfg.federate.client_num}")
    #
    #     init_cfg.expname = f"{init_cfg.model.actions}_{init_cfg.dataloader.batch_size}" \
    #                        f"_{init_cfg.train.optimizer.lr}_{init_cfg.federate.total_round_num}_{init_cfg.train.local_update_steps}"
    #     res_dir = os.path.join(init_cfg.results_DIR, init_cfg.expname)
    #     avg_res_dir = os.path.join(res_dir, 'avg_res.log')
    #     count = update_logger(init_cfg, clear_before_add=True)
    #     # get results
    #     while count < 3:
    #         init_cfg.seed = count
    #         setup_seed(init_cfg.seed)
    #         runner = get_runner(data=data,
    #                             server_class=get_server_cls(init_cfg),
    #                             client_class=get_client_cls(init_cfg),
    #                             config=init_cfg.clone(),
    #                             client_configs=client_cfgs)
    #         _ = runner.run()
    #         root_logger.info("Done!")
    #         count = update_logger(init_cfg, clear_before_add=True)
    #     #store results
    #     avg_results_str,avg_results_num = cal_path_results(os.path.join(init_cfg.results_DIR, init_cfg.expname))
    #     with open(avg_res_dir, 'a') as f:
    #         json.dump(avg_results_str, f)
    #         f.write('\n')
    #     avg_acc = avg_results_num['acc']
    #     _time = time.time() - begin
    #     print(f"time long:{_time}\n")
    #     avg_results_str['time'] = _time
    #     with open(all_res_dir, 'a') as f:
    #         f.write('\n')
    #         json.dump(f'{iteration}:{avg_results_str}', f)
    #         f.write('\n')
    #         # backward
    #     loss = model.get_loss(dummy_code, supermask, avg_acc)
    #     optimizer.zero_grad()
    #     loss.backward()
    #     optimizer.step()






















