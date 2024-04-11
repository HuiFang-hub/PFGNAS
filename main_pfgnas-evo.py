# -*- coding: utf-8 -*-
# @Time    : 2023/10/19 10:04
# @Function:
import json
import os
os.environ["AUTOGL_BACKEND"] = "pyg"
import sys
import logging
from lib.extarct_final_result import get_client_server_results
from src.GPT4GNAS.prompt.get_prompt import  prefix_prompt
from src.GPT4GNAS.prompt.get_model_list import random_generate_model_lists
import json
from src.GPT4GNAS.prompt.get_prompt import prefix_prompt
from lib.extarct_final_result import pf_extarct_all,sort_dicts_by_test_metrics
DEV_MODE = False  # simplify the src.federatedscope re-setup everytime we change
# the source codes of src.federatedscope
if DEV_MODE:
    file_dir = os.path.join(os.path.dirname(__file__), '..')
    sys.path.append(file_dir)
import random
from src.federatedscope.core.cmd_args import parse_args, parse_client_cfg
from src.federatedscope.core.auxiliaries.data_builder import get_data
from src.federatedscope.core.auxiliaries.utils import setup_seed
from src.federatedscope.core.auxiliaries.logging import update_logger
from src.federatedscope.core.auxiliaries.worker_builder import get_client_cls, \
    get_server_cls
from src.federatedscope.core.configs.config import global_cfg, CfgNode
from src.federatedscope.core.auxiliaries.runner_builder import get_runner
from lib.process_path import process_llm_path
if os.environ.get('https_proxy'):
    del os.environ['https_proxy']
if os.environ.get('http_proxy'):
    del os.environ['http_proxy']
root_logger = logging.getLogger("src.federatedscope")

# def process_path(init_cfg):
#     # gpt4 result dir
#     if init_cfg.response_DIR == "":
#         init_cfg.response_dir = os.path.join(init_cfg.response_DIR,
#                                              f"{init_cfg.federate.method}_{init_cfg.model.type}_{init_cfg.data.type}_{init_cfg.federate.client_num}")
#     else:
#         init_cfg.response_dir = os.path.join("exp",
#                                              f"{init_cfg.federate.method}_{init_cfg.model.type}_{init_cfg.data.type}_{init_cfg.federate.client_num}")
#     if not os.path.exists(init_cfg.response_dir):
#         os.makedirs(init_cfg.response_dir)
#
#     message_folder = os.path.join(init_cfg.response_dir, 'message')
#     response_folder = os.path.join(init_cfg.response_dir, 'response')
#     if not os.path.exists(message_folder):
#         os.makedirs(message_folder)
#     if not os.path.exists(response_folder):
#         os.makedirs(response_folder)
#     return init_cfg,message_folder,response_folder

if __name__ == '__main__':
    # init config
    init_cfg = global_cfg.clone()
    args = parse_args()
    if args.cfg_file:
        init_cfg.merge_from_file(args.cfg_file)
    cfg_opt, client_cfg_opt = parse_client_cfg(args.opts)
    init_cfg.merge_from_list(cfg_opt)

    init_cfg,message_folder,response_folder = process_llm_path(init_cfg)

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


    # init prompt
    system_content = prefix_prompt() # emphase key sign: "##"
    # messages_history = []
    iterations = 15
    models_res_list = []
    path = init_cfg.results_DIR
    top5_new_model = []
    for iteration in range(15):
        models_res = {}
        if top5_new_model:
            model_lists = random_generate_model_lists(init_cfg.federate.client_num, ge_n=5)
            model_lists += top5_new_model
        else:
            model_lists = random_generate_model_lists(init_cfg.federate.client_num,ge_n=10)
        # model_lists = ['hee5-ice2-jhf2-jce1-hfj2-hfj2-iii5-jce1-fbd5-ice2']
        for i,models in enumerate(model_lists):
            # operations_list = models.split('-')#operations =['abcd','efgh','ijab']
            init_cfg.model.operations = models

            response_path = os.path.join(init_cfg.response_dir, 'response.log') # store a operations combination
            with open(response_path, 'a') as f:
                json.dump(models, f)
                f.write('\n')
                if i == len(model_lists)-1: # Add another blank line
                    f.write('\n')

            print(f'{i}:{init_cfg.model.operations}')
            init_cfg.data.root = os.path.abspath(init_cfg.data.root)
            if init_cfg.data.splitter == 'lda':
                alpha = init_cfg.data.splitter_args[0]['alpha']
                init_cfg.results_DIR = os.path.join(f'{path}',
                                                    f"{init_cfg.federate.method}_pfgnas_{init_cfg.data.type}_{init_cfg.federate.client_num}_{alpha}")
            else:
                init_cfg.results_DIR = os.path.join(f'{path}',
                                                    f"{init_cfg.federate.method}_pfgnas_{init_cfg.data.type}_{init_cfg.federate.client_num}")


            init_cfg.expname = f"{init_cfg.model.operations}" \
                               f"_{init_cfg.dataloader.batch_size}" \
                               f"_{init_cfg.train.optimizer.lr}_{init_cfg.federate.total_round_num}_{init_cfg.train.local_update_steps}"
            res_dir = os.path.join(init_cfg.results_DIR, init_cfg.expname)
            avg_res_dir = os.path.join(res_dir, 'avg_res.log')
            if os.path.exists(avg_res_dir):
                with open(avg_res_dir, 'r') as file:
                    lines = file.readlines()
                    model_res = json.loads(lines[-1] )
            else:
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
            model_res = get_client_server_results(os.path.join(init_cfg.results_DIR, init_cfg.expname))
            with open(avg_res_dir, 'a') as f:
                json.dump(model_res, f)
                f.write('\n')
            models_res[init_cfg.model.operations] = model_res
        models_res_list.append(models_res)
        with open(os.path.join(init_cfg.response_dir, 'models_res_list.log'), 'a') as f:
            json.dump(models_res, f)
            f.write('\n')

        all_dicts,duplicates_set = pf_extarct_all(models_res_list)
        sorted_all_dict = sort_dicts_by_test_metrics(all_dicts, reverse=True)
        top5_keys = list( sorted_all_dict.keys())[:5]
        model_name_list = []
        for item in top5_keys:
            model_name_list += item.split('-')

        # 随机组合成包含三个新字符串的列表
        top5_new_model = ['-'.join(random.sample(model_name_list, init_cfg.federate.client_num)) for _ in range(5)]







