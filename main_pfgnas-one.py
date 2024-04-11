# -*- coding: utf-8 -*-
# @Time    : 2023/10/19 10:04
# @Function:
import json
import requests
import os
os.environ["AUTOGL_BACKEND"] = "pyg"
import sys
import logging
from lib.extarct_final_result import extarct_res, cal_path_results,get_client_server_results
from src.GPT4GNAS.prompt.get_prompt import main_prompt_word, prefix_prompt
import json
import re
from src.NAS.module_builder import get_strcut_dict, get_model_name_short
from src.GPT4GNAS.prompt.get_prompt import prefix_prompt
from src.GPT4GNAS.utils import init_llm
import ast
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
from lib.process_path import process_llm_path
if os.environ.get('https_proxy'):
    del os.environ['https_proxy']
if os.environ.get('http_proxy'):
    del os.environ['http_proxy']
root_logger = logging.getLogger("src.federatedscope")


def get_model_lists_str(iteration, init_cfg, system_content, messages_dir, response_dir):
    res = get_llm_results(iteration, init_cfg, system_content, messages_dir, response_dir)
    # process response
    res_temp = res['choices'][0]['message']['content']
    pattern = re.compile(r'\d+: (\[.*?\])')
    matches = pattern.findall(res_temp)

    input_list = [eval(match) for match in matches]
    mapping_dict = {'gcn': 'a', 'sage': 'b', 'gpr': 'c', 'gat': 'd', 'gin': 'e',
                    'fc': 'f', 'sgc': 'g', 'arma': 'h', 'appnp': 'i', 'identity': 'j',
                    'zero': 'k', 'sigmoid': '1', 'tanh': '2', 'relu': '3', 'linear': '4',
                    'elu': '5'}
    model_lists = convert_to_string(input_list, mapping_dict)
    return model_lists


def get_llm_results(iteration,init_cfg, system_content,messages_dir,response_dir):
    url, headers, link = init_llm(init_cfg, system_content)
    if os.path.exists(messages_dir):
        with open(messages_dir, 'r') as file:
            content = file.read()
            messages = json.loads(content)
    else:
        if iteration == 0:  # init operations
            messages = [
                {"role": "system", "content": system_content},
                {"role": "user",
                 "content": main_prompt_word(cfg=init_cfg, struct_dict=get_strcut_dict(), link=link, stage=iteration)}]
        else:  # update operations
            messages = [
                {"role": "system", "content": system_content},
                {"role": "user", "content": main_prompt_word(cfg=init_cfg, struct_dict=get_strcut_dict(),
                                                             link=link, models_res_list=models_res_list,
                                                             stage=iteration)},
            ]
        with open(messages_dir, 'w') as file:
            json.dump(messages, file)
    da = {
        "model": "gpt-4-1106-preview",  # "gpt-4"
        "messages": messages,
        "temperature": 0}

    response = requests.post(url, headers=headers, data=json.dumps(da))
    res = response.json()
    # messages.append(res)
    # messages_history.append(messages)

    # if os.path.exists(messages_dir):
    with open(response_dir, 'w') as file:
        json.dump(res, file)
    return res


def convert_to_string(input_list, mapping_dict):
    output_list = []
    for sublist in input_list:
        converted_sublist = ''.join(mapping_dict[node] for node in sublist)
        output_list.append(converted_sublist)
    return output_list


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

    if os.path.exists(os.path.join(init_cfg.response_dir, 'models_res_list.log')):
        models_res_list_path = os.path.join(init_cfg.response_dir, 'models_res_list.log')
        with open(models_res_list_path, 'r') as f:
            models_res_data = f.readlines()
        models_res_list = [json.loads(line.strip()) for line in models_res_data]
    else:
        models_res_list = []
    path = init_cfg.results_DIR
    for iteration in range(len(models_res_list),iterations):
        messages_dir = os.path.join(message_folder, f'message-{iteration}.log')
        response_dir = os.path.join(response_folder, f'response-{iteration}.log')
        if iteration == len(models_res_list) and os.path.exists(response_dir):
            with open(response_dir, 'r') as file:
                res = json.load(file)
            res_temp = res['choices'][0]['message']['content']
            pattern = re.compile(r'\d+: (\[.*?\])')
            matches = pattern.findall(res_temp)
            if matches:
                input_list = [eval(match) for match in matches]
                mapping_dict = {'gcn': 'a', 'sage': 'b', 'gpr': 'c', 'gat': 'd', 'gin': 'e',
                                'fc': 'f', 'sgc': 'g', 'arma': 'h', 'appnp': 'i', 'identity': 'j',
                                'zero': 'k', 'sigmoid': '1', 'tanh': '2', 'relu': '3', 'linear': '4',
                                'elu': '5'}
                model_lists = convert_to_string(input_list, mapping_dict)
            else:
                model_lists = get_model_lists_str(iteration, init_cfg, system_content, messages_dir, response_dir)
    
        elif os.path.exists(response_dir):
            os.remove(response_dir)
            model_lists = get_model_lists_str(iteration, init_cfg, system_content, messages_dir, response_dir)
        else:
            model_lists = get_model_lists_str(iteration, init_cfg, system_content, messages_dir, response_dir)
        # model_lists = ['hci3']
        models_res = {}
        # # print(f'model_lists:{model_lists}')# extract 10 operations combinations
        # print(model_lists )
        for i, models in enumerate(model_lists):
            init_cfg.model.operations = models
            response_path = os.path.join(init_cfg.response_dir, 'response.log')  # store a operations combination
            with open(response_path, 'a') as f:
                json.dump(models, f)
                f.write('\n')
                if i == len(model_lists) - 1:  # Add another blank line
                    f.write('\n')

            print(f'{i}:{init_cfg.model.operations}')
            init_cfg.data.root = os.path.abspath(init_cfg.data.root)
            if init_cfg.data.splitter == 'lda':
                alpha = init_cfg.data.splitter_args[0]['alpha']
                init_cfg.results_DIR = os.path.join(f'{path}',
                                                    f"{init_cfg.federate.method}_{init_cfg.model.type}_{init_cfg.data.type}_{init_cfg.federate.client_num}_{alpha}")
            else:
                init_cfg.results_DIR = os.path.join(f'{path}',
                                                    f"{init_cfg.federate.method}_{init_cfg.model.type}_{init_cfg.data.type}_{init_cfg.federate.client_num}")
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

            model_res, _ = cal_path_results(os.path.join(init_cfg.results_DIR, init_cfg.expname))
            with open(avg_res_dir, 'a') as f:
                json.dump(model_res, f)
                f.write('\n')
            models_res[init_cfg.model.operations] = model_res
        models_res_list.append(models_res)
        with open(os.path.join(init_cfg.response_dir, 'models_res_list.log'), 'a') as f:
            json.dump(models_res, f)
            f.write('\n')



        # input_lst = res_temp.split('model:')
        # for i in range(1, len(input_lst)):
        #     #extract opreation
        #     operations_str = input_lst[i].split('[')[1].split(']')[0]
        #     operations_list = operations_str.split(',')
        #     operations_list_str = [a.replace(" ", "") for a in operations_list]
            # compute results accoding to those operations
           #  '''
           # models_res_list = [{'abcd':{"acc":0.1233,"roc_auc":0.2333},'cdff':{'acc':0.4354,'roc_auc':0.9843},
           #              'dfed':{'acc':0.4354,'roc_auc':0.4875},'abdc':{'acc':0.3671,'roc_auc':0.8940},
           #              'bcda':{'acc':0.4258,'roc_auc':0.1238},'fcab':{'acc':0.1584,'roc_auc':0.2594},
           #              'aaaa':{'acc':0.9215,'roc_auc':0.6127},'bfda':{'acc':0.1565,'roc_auc':0.0534},
           #              'aeef':{'acc':0.4354,'roc_auc':0.9587},'edbb':{'acc':0.1259,'roc_auc':0.3207}},
           #             {'bcda': {"acc": 0.2786, "roc_auc": 2855}, 'cdfa': {'acc': 0.1525, 'roc_auc': 0.2687},
           #              'efba': {'acc': 0.4354, 'roc_auc': 0.4875}, 'ffdc': {'acc': 0.2561, 'roc_auc': 0.3856},
           #              'edcb': {'acc': 0.1575, 'roc_auc': 0.5377}, 'bfab': {'acc': 0.1584, 'roc_auc': 0.2568},
           #              'aaaa': {'acc': 0.3584, 'roc_auc': 0.4686}, 'bbbb': {'acc': 0.3968, 'roc_auc': 0.9687},
           #              'eeee': {'acc': 0.3434, 'roc_auc': 0.4468}, 'cccc': {'acc': 0.2534, 'roc_auc': 0.6867}},
           #             ]
           #  '''

            # adjust style

        # input the result to llm




