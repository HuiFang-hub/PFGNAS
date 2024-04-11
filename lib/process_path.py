# -*- coding: utf-8 -*-
# @Time    : 2024/1/5 13:23
# @Function:
import os

def process_llm_path(init_cfg):
    if init_cfg.data.splitter == 'lda':
        alpha = init_cfg.data.splitter_args[0]['alpha']
        init_cfg.response_dir = os.path.join(init_cfg.response_DIR,
                                                 f"{init_cfg.federate.method}_{init_cfg.model.type}_{init_cfg.data.type}_{alpha}_{init_cfg.federate.client_num}")
    else:
        init_cfg.response_dir = os.path.join(init_cfg.response_DIR,
                                                 f"{init_cfg.federate.method}_{init_cfg.model.type}_{init_cfg.data.type}_{init_cfg.federate.client_num}")
    if not os.path.exists(init_cfg.response_dir):
        os.makedirs(init_cfg.response_dir)

    message_folder = os.path.join(init_cfg.response_dir, 'message')
    response_folder = os.path.join(init_cfg.response_dir, 'response')
    if not os.path.exists(message_folder):
        os.makedirs(message_folder)
    if not os.path.exists(response_folder):
        os.makedirs(response_folder)
    return init_cfg, message_folder, response_folder



def process_path(init_cfg):
    if init_cfg.data.splitter == 'lda':
        alpha = init_cfg.data.splitter_args[0]['alpha']
        init_cfg.response_dir = os.path.join(init_cfg.response_DIR,
                                                 f"{init_cfg.federate.method}_{init_cfg.model.type}_{init_cfg.data.type}_{alpha}_{init_cfg.federate.client_num}")
    else:
        init_cfg.response_dir = os.path.join(init_cfg.response_DIR,
                                                 f"{init_cfg.federate.method}_{init_cfg.model.type}_{init_cfg.data.type}_{init_cfg.federate.client_num}")
    #  result dir
    # if init_cfg.response_DIR == "":
    #     init_cfg.response_dir = os.path.join(init_cfg.response_DIR,
    #                                          f"{init_cfg.federate.method}_{init_cfg.model.type}_{init_cfg.data.type}_{init_cfg.federate.client_num}")
    # else:
    #     init_cfg.response_dir = os.path.join("exp",
    #                                          f"{init_cfg.federate.method}_{init_cfg.model.type}_{init_cfg.data.type}_{init_cfg.federate.client_num}")
    if not os.path.exists(init_cfg.response_dir):
        os.makedirs(init_cfg.response_dir)
    return init_cfg