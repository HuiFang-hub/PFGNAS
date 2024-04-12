import os
import json
import re
import numpy as np
def get_res_path(baseline, gnn_types,data_name, client,federate_method='FedAvg') :
    # get res path
    if baseline in gnn_types:
        directory = f'results/{federate_method}_{baseline}_{data_name}_{client}/200_0.005_100_4'
        
    elif baseline in ['ditto','FedEM']:
        directory = f'results/{baseline}_gat_{data_name}_{client}'
        
    elif baseline in ['FedPUB']:
        directory = f'results/{baseline}_unify_data_{data_name}_{client}/200_0.01_100_4'
        
    elif baseline == 'darts':
        directory = f'results/{baseline}_fl-{baseline}_{data_name}_{client}'
        
    elif baseline in ['pfgnas-random', 'pfgnas-evo','pfgnas-one', 'pfgnas']:
        directory = f'exp/{federate_method}_{baseline}_{data_name}_{client}'
        
    else:
        directory = f'results/{federate_method}_{baseline}_{data_name}_{client}'
        
    if not os.path.exists(directory):
        print(f"路径 '{directory}' 不存在")
        exit
    return directory

def find_file(filename, search_path):
    for root, dirs, files in os.walk(search_path):
        if filename in files:
            return os.path.join(root, filename)
    print(f"路径 '{os.path.join(search_path, filename)}' 不存在")
    exit

def get_merged_dict(directory):
    # find res
    file_path = find_file('models_res_list.log', directory)
    with open(file_path, 'r') as f:
        data = f.readlines()
    models_res_list = [json.loads(line.strip()) for line in data]
    
    # Merge and Remove Duplicates 
    merged_dict = {}
    for item in models_res_list:
        for key, value in item.items():
            merged_dict.setdefault(key, {}).update(value)
    return merged_dict

def unify_metric(metric):
    if metric == 'test_acc' or metric == 'acc':
        metric='Accuracy'
    elif metric=='test_roc_auc' or metric == 'roc_auc':
        metric = 'AUC'
    return metric

def get_best_res(max_acc_value,res_df,baseline):
    for key, value_string in max_acc_value.items():
        key = unify_metric(key)
        if isinstance(value_string, str) and "±" in value_string:
            value, error = [f"{float(part) * 100:.2f}" for part in value_string.split("±")]
            res_df.loc[baseline, key] = f"{value}±{error}"
        else:
            res_df.loc[baseline, key] = value_string
    
    return res_df

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


def replace_key(res_dict):
    new_dict = {}
    for key, value in res_dict.items():
        new_key = {k.replace('test_', ''): v for k, v in value.items()}
        client_key = 'Global' if key=='0' else key
        new_dict[client_key] = new_key
    return new_dict

def extract_results(raw_results):
    raw_results = raw_results.replace("'", "\"")  # Replacing single quotes with double quotes for valid JSON
    results_dict = eval(raw_results)  # Using eval to convert the string to a dictionary

    # Extract only the required keys
    keys_to_extract = ['train_acc', 'test_acc', 'train_roc_auc', 'test_roc_auc']
    filtered_results = {key: results_dict[key] for key in keys_to_extract if key in results_dict}

    return filtered_results