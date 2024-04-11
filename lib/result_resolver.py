import copy
import json
import os
import re

import numpy as np
def extract_client_fedpub(file_path):
    with open(file_path, 'r') as f:
        res_data = f.readlines()
    last_line = res_data[-1].strip()
    client_dict = json.loads(last_line)['log']
    last_values_dict = {}
    for metric in client_dict.keys():
        if metric in ['rnd_test_acc', 'rnd_test_roc_auc','rnd_local_test_acc', 'rnd_local_test_roc_auc']:
            new_metric = 'acc' if 'acc' in metric else 'roc_auc'
            last_values_dict[new_metric] = client_dict[metric][-1]
    return last_values_dict


# def fedpub_avg_list_dict(res_list):
#     for i in res_list:
#         for key, value in dict.items():

# def calculate_average_dict_simple(res_list):
#     acc_totals = {key: 0 for key in res_list[0].keys()}
#     roc_auc_totals = {key: 0 for key in res_list[0].keys()}
#     num_dicts = len(res_list)
#
#     # 遍历列表中的字典
#     for item in res_list:
#         for key, values in item.items():
#             acc_totals[key] += values['acc']
#             roc_auc_totals[key] += values['roc_auc']
#
#     # 计算均值
#     avg_acc = {key: total / num_dicts for key, total in acc_totals.items()}
#     avg_roc_auc = {key: total / num_dicts for key, total in roc_auc_totals.items()}
#
#     res_avg_dict= {'acc':avg_acc,'roc_auc':avg_roc_auc}
#     return  res_avg_dict


# def calculate_average_dict_simple(res_list):
#     acc_totals = {key: 0 for key in res_list[0].keys()}
#     # acc_totals = {str('Global' if key== '0' else str(int(key)+1)): 0 for key in res_list[0].keys()}
#     roc_auc_totals = {key: 0 for key in res_list[0].keys()}
#     num_dicts = len(res_list)
#
#     # 遍历列表中的字典
#     for item in res_list:
#         for key, values in item.items():
#             acc_totals[key] += values['acc']
#             roc_auc_totals[key] += values['roc_auc']
#
#     # 计算均值
#     avg_acc = {key: total / num_dicts for key, total in acc_totals.items()}
#     avg_roc_auc = {key: total / num_dicts for key, total in roc_auc_totals.items()}
#
#     res_avg_dict= {'acc':avg_acc,'roc_auc':avg_roc_auc}
#     return  res_avg_dict


def calculate_average_dict_with_variance(res_list):
    acc_totals = {key: [] for key in res_list[0].keys()}
    roc_auc_totals = {key: [] for key in res_list[0].keys()}
    num_dicts = len(res_list)

    # 遍历列表中的字典
    for item in res_list:
        for key, values in item.items():
            acc_totals[key].append(values['acc'])
            roc_auc_totals[key].append(values['roc_auc'])

    # 计算均值和方差
    avg_acc = {key: np.mean(values) for key, values in acc_totals.items()}
    var_acc = {key: np.var(values) for key, values in acc_totals.items()}

    avg_roc_auc = {key: np.mean(values) for key, values in roc_auc_totals.items()}
    var_roc_auc = {key: np.var(values) for key, values in roc_auc_totals.items()}

    # 使用 '±' 连接均值和方差
    res_avg_dict = {
        'acc': {key: f"{avg_acc[key]*100:.2f}" for key in avg_acc},
        'roc_auc': {key: f"{avg_roc_auc[key]*100:.2f}" for key in avg_roc_auc}
    }

    res_avg_var_dict = {
        'acc': {key: f"{avg_acc[key]*100:.2f}±{var_acc[key]*100:.2f}" for key in avg_acc},
        'roc_auc': {key: f"{avg_roc_auc[key]*100:.2f}±{var_roc_auc[key]*100:.2f}" for key in avg_roc_auc}
    }

    return res_avg_dict,res_avg_var_dict


def add_v(res_dict,original_res,r,new_baseline_name):
    res_dict['Round'] += [r] * len(original_res['acc'])
    res_dict['Methods'] += [new_baseline_name] * len(original_res['acc'])
    res_dict['Accuracy'] += original_res['acc']
    res_dict['ROC_AUC'] += original_res['roc_auc']
    return res_dict

def get_original_res(path):  # max_acc_key = 'gdc3-ebh5-afi1'

    model_results = []
    for root, dirs, files in os.walk(path):
        for dir_name in dirs[:]:
            if dir_name != "model":  # 使用切片复制dirs列表以便在循环中修改它
                dir_path = os.path.join(root, dir_name)
                best_res_log_path = os.path.join(dir_path, "eval_results_raw.log")
                if os.path.exists(best_res_log_path):
                    one_res_dict = extarct_client_server_res( best_res_log_path)  # {'2': {'train_acc': 0.7204, 'train_roc_auc': 0.918}, '1': {'train_acc': 0.8225, 'train_roc_auc': 0.9561}, '3': {'train_acc': 0.4932, 'train_roc_auc': 0.9021}, '0': {'test_acc': 0.5622, 'test_roc_auc': 0.6593}}
                    new_dict = replace_key(one_res_dict)
                    # for key, value in one_res_dict.items():
                    #     new_key = {k.replace('test_', ''): v for k, v in value.items()}
                    #     new_dict[key] = new_key
                    model_results.append( new_dict)
    original_res = {'acc':[],'roc_auc':[]}
    for res in  model_results:
        res = res['Global']
        original_res['acc'].append(res['acc'])
        original_res['roc_auc'].append(res['roc_auc'])
    return original_res,model_results


def extarct_client_server_res(file):
    if os.path.exists(file):
        # read log file
        with open(file, 'r') as f:
            log_content = f.read()
            text = log_content.replace('nan', '0.5')
    else:
        text = file

    client_pattern = re.compile(
        r"'Role': 'Client #(\d+)', 'Round': (\d+), 'Results_raw': ({.*?})")

    server_pattern = re.compile(
        r"'Role': 'Server #', 'Round': 'Final', 'Results_raw': {'server_global_eval': ({.*?})}")

    results_dict = {}

    matches = client_pattern.finditer(text)
    for match in matches:
        client_id, _, raw_results = match.groups()
        results_dict[client_id] = extract_results(raw_results)

    server_match = server_pattern.search(text)
    if server_match:
        raw_results = server_match.group(1)
        results_dict['0'] = extract_results(raw_results)

    return results_dict

def extract_results(raw_results):
    raw_results = raw_results.replace("'", "\"")  # Replacing single quotes with double quotes for valid JSON
    results_dict = eval(raw_results)  # Using eval to convert the string to a dictionary

    # Extract only the required keys
    keys_to_extract = ['train_acc', 'test_acc', 'train_roc_auc', 'test_roc_auc']
    filtered_results = {key: results_dict[key] for key in keys_to_extract if key in results_dict}

    return filtered_results

def replace_key(res_dict):
    new_dict = {}
    for key, value in res_dict.items():
        new_key = {k.replace('test_', ''): v for k, v in value.items()}
        client_key = 'Global' if key=='0' else key
        new_dict[client_key] = new_key
    return new_dict

def replace_key_simple(res_dict):
    new_key = {k.replace('test_', ''): v for k, v in res_dict()}

    return new_key

def cal_round_avgAndmax_res(models_res_list,res_line_dict,max_res_line_dict,accum_max_res_line_dict,method,fig_round = None):
    if fig_round is None:
        round = len(models_res_list)
    for r in range(round):
        # print(r)
        _, avg_dict = calculate_average_dict(models_res_list[r])
        # avg_dict = replace_key_simple(avg_dict)
        avg_dict = {k.replace('test_', ''): v for k, v in avg_dict.items()}
        res_line_dict = add_v(res_line_dict,avg_dict,r,method)

        # 将两个列表打包并按照规则排序
        sorted_data = sorted(zip(avg_dict['acc'], avg_dict['roc_auc']), key=lambda x: (x[0], x[1]),
                             reverse=True)[:3]
        avg_dict = {'acc': [item[0] for item in sorted_data], 'roc_auc': [item[1] for item in sorted_data]}
        # avg_dict = {key: sorted(values, reverse=True)[:5] for key, values in avg_dict .items()}
        # _, avg_dict = calculate_average_dict_simple(result_dict)
        max_res_line_dict = add_v( max_res_line_dict, avg_dict, r, method)
        # max_res_line_dict['Accuracy'] += avg_dict['acc']
        # max_res_line_dict['ROC_AUC'] += avg_dict['roc_auc']
        # max_res_line_dict['Round'] += [r] * len(avg_dict['acc'])
        # max_res_line_dict['Methods'] += [method] * len(avg_dict['acc'])

        ##################
        merged_dict = {key: value for d in models_res_list[:r+1] for key, value in d.items()}
        _, avg_dict = calculate_average_dict(merged_dict)
        sorted_data = sorted(zip(avg_dict['test_acc'], avg_dict['test_roc_auc']), key=lambda x: (x[0], x[1]),
                             reverse=True)[:3]
        avg_dict = {'acc': [item[0] for item in sorted_data], 'roc_auc': [item[1] for item in sorted_data]}
        # avg_dict = {key: sorted(values, reverse=True)[:5] for key, values in avg_dict .items()}
        # _, avg_dict = calculate_average_dict_simple(result_dict)
        accum_max_res_line_dict= add_v(accum_max_res_line_dict,avg_dict,r,method)

        # accum_max_res_line_dict['Accuracy'] += avg_dict['acc']
        # accum_max_res_line_dict['ROC_AUC'] += avg_dict['roc_auc']
        # accum_max_res_line_dict['Round'] += [r] * len(avg_dict['acc'])
        # accum_max_res_line_dict['Methods'] += [method] * len(avg_dict['acc'])

    return  res_line_dict,max_res_line_dict,accum_max_res_line_dict


def calculate_average_dict(dict_A):
    # 初始化平均值字典
    average_dict = {}

    # 遍历字典A，计算平均值
    for _, value_outer in dict_A.items():
        for key_inner, value_inner in value_outer.items():
            if key_inner not in average_dict:
                average_dict[key_inner] = {}
            for sub_key, sub_value in value_inner.items():
                if isinstance(sub_value, str):
                    val = sub_value.split('±')
                    sub_value = float(val[0])
                average_dict[key_inner].setdefault(sub_key, []).append(sub_value)
    ori_avg_list = copy.deepcopy(average_dict['0'])
    # 计算平均值
    for key_inner, value_inner in average_dict.items():
        for sub_key, sub_values in value_inner.items():
            average_dict[key_inner][sub_key] = round(sum(sub_values) / len(sub_values),5)

    return average_dict,ori_avg_list


def calculate_average_dict_pgnasone(data):
    # 初始化平均值字典
    acc_values = []
    roc_auc_values = []

    for values in data.values():
        for metric, value in values.items():
            if metric == 'acc':
                mean_acc, _ = map(float, value.split('±'))
                acc_values.append(mean_acc)
            elif metric == 'roc_auc':
                mean_roc_auc, _ = map(float, value.split('±'))
                roc_auc_values.append(mean_roc_auc)

    res_avg_dict = {'acc': acc_values, 'roc_auc': roc_auc_values}
    return res_avg_dict

    # return average_dict,ori_avg_list