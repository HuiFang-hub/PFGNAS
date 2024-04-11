  # -*- coding: utf-8 -*-
# @Time    : 2023/10/26 15:04
# @Function:
import json
import time

import requests
import os
import sys
import logging

from lib.extarct_final_result import extarct_res, cal_path_results
from src.FLAGNNS import utils
from src.GPT4GNAS.prompt.get_prompt import main_prompt_word, prefix_prompt
import json
import re
from src.NAS.module_builder import get_strcut_dict, get_model_name_short
from src.GPT4GNAS.prompt.get_prompt import prefix_prompt
from src.GPT4GNAS.utils.LLM_init import init_llm
import random
DEV_MODE = False  # simplify the src.federatedscope re-setup everytime we change
# the source codes of src.federatedscope
if DEV_MODE:
    file_dir = os.path.join(os.path.dirname(__file__), '..')
    sys.path.append(file_dir)

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
def process_path(init_cfg):
    #  result dir
    if init_cfg.response_DIR == "":
        init_cfg.response_dir = os.path.join(init_cfg.response_DIR,
                                             f"{init_cfg.federate.method}_{init_cfg.model.type}_{init_cfg.data.type}_{init_cfg.federate.client_num}")
    else:
        init_cfg.response_dir = os.path.join("exp",
                                             f"{init_cfg.federate.method}_{init_cfg.model.type}_{init_cfg.data.type}_{init_cfg.federate.client_num}")
    if not os.path.exists(init_cfg.response_dir):
        os.makedirs(init_cfg.response_dir)

    # message_folder = os.path.join(init_cfg.response_dir, 'message')
    # response_folder = os.path.join(init_cfg.response_dir, 'response')
    # if not os.path.exists(message_folder):
    #     os.makedirs(message_folder)
    # if not os.path.exists(response_folder):
    #     os.makedirs(response_folder)
    return init_cfg

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


    # generate mask
    NUM_POP = 60
    SAMPLE_EPOCH = 5
    SAMPLE_SIZE = 20
    # supermasks = [utils.random_supermask() for i in range(NUM_POP)]
    iterations = 15
    # for iteration in range(iterations):
    st_time = time.time()
    for sample_epoch in range(SAMPLE_EPOCH):
        # sample_supermasks = random.sample(supermasks, SAMPLE_SIZE )
        # cal result

        # results dir
        init_cfg.results_DIR = os.path.join('results',
                                            f"{init_cfg.federate.method}_{init_cfg.model.type}_{init_cfg.data.type}_{init_cfg.federate.client_num}")
        init_cfg.expname = f"{init_cfg.model.actions}_{init_cfg.dataloader.batch_size}" \
                           f"_{init_cfg.train.optimizer.lr}_{init_cfg.federate.total_round_num}_{init_cfg.train.local_update_steps}"
        res_dir = os.path.join(init_cfg.results_DIR, init_cfg.expname)
        avg_res_dir = os.path.join(res_dir, 'avg_res.log')
        count = update_logger(init_cfg, clear_before_add=True)
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

    model_res,_ = cal_path_results(os.path.join(init_cfg.results_DIR, init_cfg.expname))
    with open(avg_res_dir, 'a') as f:
        json.dump(model_res, f)
        f.write('\n')
    # models_res[init_cfg.model.operations] = model_res






















