# -*- coding: utf-8 -*-
# @Time    : 2023/10/19 10:04
# @Function:
import ast
import requests
import os
os.environ["AUTOGL_BACKEND"] = "pyg"
import sys
import logging
from lib.extarct_final_result import get_client_server_results
from lib.process_path import process_llm_path
from src.GPT4GNAS.prompt.get_prompt import main_prompt_word, prefix_prompt
import json
import re
from src.NAS.module_builder import get_strcut_dict, get_model_name_short
from src.GPT4GNAS.prompt.get_prompt import prefix_prompt
from src.GPT4GNAS.utils import init_llm
# import google.generativeai as genai

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
from zhipuai import ZhipuAI

if os.environ.get('https_proxy'):
    del os.environ['https_proxy']
if os.environ.get('http_proxy'):
    del os.environ['http_proxy']
root_logger = logging.getLogger("src.federatedscope")

# import debugpy
# try:
#     debugpy.listen(("localhost", 9502))
#     print("Waiting for debugger attach")
#     debugpy.wait_for_client()
# except Exception as e:
#     pass


def process_llm_path(init_cfg):
    if init_cfg.data.splitter == 'lda':
        alpha = init_cfg.data.splitter_args[0]['alpha']
        init_cfg.response_dir = os.path.join(init_cfg.response_DIR,
                                                 f"{init_cfg.federate.method}_{init_cfg.model.type}_{init_cfg.data.type}_{alpha}_{init_cfg.federate.client_num}_{init_cfg.llm.type}")
    else:
        init_cfg.response_dir = os.path.join(init_cfg.response_DIR,
                                                 f"{init_cfg.federate.method}_{init_cfg.model.type}_{init_cfg.data.type}_{init_cfg.federate.client_num}_{init_cfg.llm.type}")
    if not os.path.exists(init_cfg.response_dir):
        os.makedirs(init_cfg.response_dir)

    message_folder = os.path.join(init_cfg.response_dir, 'message')
    response_folder = os.path.join(init_cfg.response_dir, 'response')
    if not os.path.exists(message_folder):
        os.makedirs(message_folder)
    if not os.path.exists(response_folder):
        os.makedirs(response_folder)
    return init_cfg, message_folder, response_folder


def get_llm_results(iteration,init_cfg, system_content,messages_dir,response_dir,models_res_list):
    # get your API_key
    if 'gpt' in init_cfg.llm.type:
        api_key, url= 'get your api_key', 'get your url'
        headers = init_llm(api_key)
  
    if os.path.exists(messages_dir):
        with open(messages_dir, 'r') as file:
            content = file.read()
            messages = json.loads(content)
    else:
        if iteration == 0:  # init operations
            models_res_list = None 
            
            # update operations
        if 'gpt' in init_cfg.llm.type or 'glm' in init_cfg.llm.type:
            messages = [
                {"role": "system", "content": system_content},
                {"role": "user", "content": main_prompt_word(cfg=init_cfg, struct_dict=get_strcut_dict(),
                                                            link=None, models_res_list=models_res_list,
                                                            stage=iteration)},
            ]
        elif 'palm' in init_cfg.llm.type:
            messages = main_prompt_word(cfg=init_cfg, struct_dict=get_strcut_dict(),
                                                            link=None, models_res_list=models_res_list,
                                                            stage=iteration)
            
        with open(messages_dir, 'w') as file:
            json.dump(messages, file)
    # initial LLM
    if 'gpt' in init_cfg.llm.type:
       
        da = {
            "model": "gpt-4-1106-preview",  # "gpt-4"
            "messages": messages,
            "temperature": 0.5}

        response = requests.post(url, headers=headers, data=json.dumps(da))
        res = response.json()
        with open(response_dir, 'w') as file:
            json.dump(res, file)
   
    elif 'palm' in init_cfg.llm.type:
        #https://colab.research.google.com/drive/1L-cfen2dQeCyyz-H_tAN6kJt00a0hJit#scrollTo=ZGqQUNljxT0q      
        with open(response_dir, 'r') as file:
            res = file.read()
        if res is None or res == "":
            print("waiting for the result from palm!")
            exit()    
    elif 'glm' in init_cfg.llm.type:
        client = ZhipuAI(api_key="your api key")
        response = client.chat.completions.create(
                model="glm-4",  # model name
                messages=messages,)      
        res = response.choices[0].message.content 
        with open('log.txt', 'w') as file:
            file.write(res)  

    return res

def convert_to_string(sub_list,mapping_dict):
    return '-'.join(''.join(mapping_dict[op] for op in sublist) for sublist in sub_list)

def get_model_lists_str(iteration, init_cfg, system_content, messages_dir, response_dir,models_res_list):
    res = get_llm_results(iteration, init_cfg, system_content, messages_dir, response_dir,models_res_list)

    if 'gpt' in init_cfg.llm.type:
        res = res['choices'][0]['message']['content']

    pattern = r'\d+: \[\[(.*?)\]\]'
    matches = re.findall(pattern, res, re.DOTALL)
    model_lists = [ast.literal_eval('[[' + match + ']]') for match in matches]
    mapping_dict = {'gcn': 'a', 'sage': 'b', 'gpr': 'c', 'gat': 'd', 'gin': 'e',
                    'fc': 'f', 'sgc': 'g', 'arma': 'h', 'appnp': 'i', 'identity': 'j',
                    'zero': 'k', 'sigmoid': '1', 'tanh': '2', 'relu': '3', 'linear': '4',
                    'elu': '5'}

    model_lists_str = [convert_to_string(sub_list,mapping_dict) for sub_list in model_lists]
    return model_lists_str


if __name__ == '__main__':
    ########### init config
    init_cfg = global_cfg.clone()
    args = parse_args()
    if args.cfg_file:
        init_cfg.merge_from_file(args.cfg_file)
    cfg_opt, client_cfg_opt = parse_client_cfg(args.opts)
    init_cfg.merge_from_list(cfg_opt)
    init_cfg,message_folder,response_folder = process_llm_path(init_cfg)
    ############ if client_cfg, load it
    if args.client_cfg_file:
        client_cfgs = CfgNode.load_cfg(open(args.client_cfg_file, 'r'))
        client_cfgs.merge_from_list(client_cfg_opt)
    else:
        client_cfgs = None

    ############## data config
    data, modified_cfg = get_data(config=init_cfg.clone(),
                                  client_cfgs=client_cfgs)
    init_cfg.merge_from_other_cfg(modified_cfg)

    ############## init prompt
    system_content = prefix_prompt() # emphase key sign: "##"

    ############## check for 'models_res_list.log', and get models_list
    if os.path.exists(os.path.join(init_cfg.response_dir, 'models_res_list.log')):
        models_res_list_path = os.path.join(init_cfg.response_dir, 'models_res_list.log')
        with open(models_res_list_path, 'r') as f:
            models_res_data = f.readlines()
        models_res_list = [json.loads(line.strip()) for line in models_res_data]
    else:
        models_res_list = []
    path= init_cfg.results_DIR
    iterations = 15
    for iteration in range(len(models_res_list), iterations):
        messages_dir = os.path.join(message_folder, f'message-{iteration}.log')
        response_dir = os.path.join(response_folder, f'response-{iteration}.log')

        ############## decode the str of model_list
        if iteration == len(models_res_list) and  os.path.exists(response_dir):
            if 'gpt' in init_cfg.llm.type:
                with open(response_dir, 'r') as file:
                    res = json.load(file)
                res_temp = res['choices'][0]['message']['content']
        
            elif 'palm' in init_cfg.llm.type or 'glm' in init_cfg.llm.type:
                with open(response_dir, 'r') as file:
                    res_temp = file.read() 

            pattern = r'\d+: \[\[(.*?)\]\]'
            matches = re.findall(pattern, res_temp, re.DOTALL)
            if matches:
                model_lists = [ast.literal_eval('[[' + match + ']]') for match in matches]
                mapping_dict = {'gcn': 'a', 'sage': 'b', 'gpr': 'c', 'gat': 'd', 'gin': 'e',
                                'fc': 'f', 'sgc': 'g', 'arma': 'h', 'appnp': 'i', 'identity': 'j',
                                'zero': 'k', 'sigmoid': '1', 'tanh': '2', 'relu': '3', 'linear': '4',
                                'elu': '5'}
                model_lists_str = [convert_to_string(sub_list,mapping_dict) for sub_list in model_lists]
            else:
                model_lists_str = get_model_lists_str(iteration, init_cfg, system_content, messages_dir, response_dir,models_res_list)
        elif os.path.exists(response_dir):
            os.remove(response_dir)
            model_lists_str = get_model_lists_str(iteration, init_cfg, system_content, messages_dir, response_dir,models_res_list)
        else:
            model_lists_str = get_model_lists_str(iteration, init_cfg, system_content, messages_dir, response_dir,models_res_list)
        print(model_lists_str ) #['hdi1-jjj2-ibb1-jdb4-fkj2',...]
        if not model_lists_str:
            print("List is empty. Exiting program.")
            exit()
        models_res = {}
        
        ########### federated training
        for i, models in enumerate(model_lists_str):
            init_cfg.model.operations = models
            response_path = os.path.join(init_cfg.response_dir, 'response.log')  # store a operations combination
            with open(response_path, 'a') as f:
                json.dump(models, f)
                f.write('\n')
                if i == len(model_lists_str) - 1:  # Add another blank line
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

                ############## get performance
                model_res = get_client_server_results(os.path.join(init_cfg.results_DIR, init_cfg.expname))
                with open(avg_res_dir, 'a') as f:
                    json.dump(model_res, f)
                    f.write('\n')
            models_res[init_cfg.model.operations] = model_res
        models_res_list.append(models_res)
        ############## save the performance and waiting for next iteration
        with open(os.path.join(init_cfg.response_dir, 'models_res_list.log'), 'a') as f:
            json.dump(models_res, f)
            f.write('\n')




