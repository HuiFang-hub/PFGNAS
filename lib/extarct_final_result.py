import copy
import glob
import re
import ast
import os
import sys

current_dir = os.path.dirname(os.path.abspath(__file__))
root_path = os.path.dirname(current_dir)
sys.path.append(root_path)
import numpy as np
import pandas as pd
from statistics import mean, stdev
import json
import matplotlib.pyplot as plt

from lib.result_resolver import extract_client_fedpub, calculate_average_dict_with_variance, add_v, get_original_res, \
    cal_round_avgAndmax_res, extarct_client_server_res, calculate_average_dict, calculate_average_dict_pgnasone
from lib.visual import plot_Grouped_barplots, plot_line_bond, plot_radar_chart
import os

def extarct_res(file):
    if os.path.exists(file):
    # read log file
        with open(file, 'r') as f:
            log_content = f.read()
            text = log_content.replace('nan', '0.5')
    else:
        text = file
    test_m = ['acc', 'roc_auc', 'loss']
    pattern = r"'Results_raw': {'server_global_eval': ({.*?})}"
    matches = re.findall(pattern, text)
    res = ast.literal_eval(matches[0])
    values = {}
    for m in test_m:
        values[m] = res[f'test_{m}']
    return values



def extarct_metric(res,metrics):
    avg_dict = {}
    for metric in metrics:
        r_list = []
        for info in res.values():
            if metric in info:
                if isinstance(info[metric], str):
                    val = info[metric].split('±')
                    r_list.append(float(val[0]))
                else:
                    r_list.append(info[metric])
        avg_dict[metric] = round(sum( r_list) / len( r_list) if  r_list else 0, 4)
    return  avg_dict


def extarct_all(models_res_list,metrics):
    all_dict = {}
    all_operations = []
    duplicate_keys = set()
    for one_round_resuts in models_res_list:
        all_operations += list(one_round_resuts.keys())
        for one_round_key, one_round_res in one_round_resuts.items():
            if one_round_key in all_dict:
                duplicate_keys.add(one_round_key)
            all_dict[one_round_key] = {}
            for m in metrics:
                all_dict[one_round_key][m]= one_round_res[m]

    # duplicates_list = list(set([item for item in all_operations if all_operations.count(item) > 1]))
    return all_dict,duplicate_keys

def pf_extarct_all(models_res_list):
    merged_dict = {}
    duplicate_keys = set()

    for d in models_res_list:
        for key, value in d.items():
            if key in merged_dict:
                duplicate_keys.add(key)
            merged_dict[key] = value

    return merged_dict, duplicate_keys

def extarct_repeat_operations(models_res_list):
    all_operations = []
    for one_round_resuts in models_res_list:
        all_operations += list(one_round_resuts.keys())
    duplicates = list(set([item for item in all_operations if all_operations.count(item) > 1]))
    return duplicates


def cal_path_results(path):
    model_results = []
    metric = None
    for root, dirs, files in os.walk(path):
        for dir_name in dirs[:]:
            if dir_name != "model": 
                dir_path = os.path.join(root, dir_name)
                best_res_log_path = os.path.join(dir_path, "eval_results.log")
               
                if os.path.exists(best_res_log_path):
                    model_res = extarct_res(best_res_log_path)
                    metric = list(model_res.keys())
                    model_results.append(model_res)  #'±'
    # avg
    avg_results_str,avg_results_num = avg_res(model_results,metric)
    return avg_results_str,avg_results_num

def avg_res(model_results,metric):
    all_res = {key: [] for key in metric}
    for m_dict in model_results:
        for m, value in m_dict.items():
            all_res[m].append(value)
    avg_results_str = {}
    avg_results_num = {}
    for k, v in all_res.items():
        mean = np.mean(v)
        stddev = np.std(v)
        avg_results_num[k] = round(mean, 4)
        avg_results_str[k]= str(round(mean,4)) + '±' + str(round(stddev,4))
    return avg_results_str,avg_results_num



    # plt.show()
def  get_offcial_name(name):
    if name in ['pfgnas-evo','pfgnas-random','pfgnas-one']:
        n = name.split('-')
        new = n[0].upper() + '-' + n[1].upper()[0]
    elif name == 'fl-graphnas':
        new = 'FL+GraphNAS'
    elif name == 'fl-agnns':
        new = 'FLAGNNS'
    elif name in ['darts']:
        new = name.capitalize() #shouzimu daxie
    elif name=='sage':
        new = 'GraphSage'
    elif name=='FedPUB':
        new = name
    else:
        new = name.upper()  #daxie
    return new

def extract_gnn_res(data_name, client, federate_method='FedAvg',splitter=None, beta=None,res_barplot_dict=None, res_line_dict=None, max_res_line_dict=None, accum_max_res_line_dict=None,fig_round = None):

    if splitter:
        out_path = f'exp_{splitter}/{data_name}_{client}'
    else:
        out_path = f'exp/{data_name}_{client}'
    if not os.path.exists( out_path ):
        os.makedirs( out_path )
    # gnn_types=['gcn', 'sage', 'gat', 'gin', 'gpr', 'sgc', 'arma', 'appnp'] # fc：None
    gnn_types = ['gcn', 'sage', 'gat', 'gin', 'sgc', 'arma', 'appnp']
    # fl_types=['ditto','FedPUB'] #['ditto','FedEM','FedPUB']
    fl_types=['FedPUB'] 
    # nas_types = ['darts', 'fl-graphnas', 'agnns','pfgnas-evo', 'pfgnas-random', 'pfgnas-one', 'pfgnas']
    nas_types = ['fl-agnns', 'pfgnas']


    if res_barplot_dict:
        # baselines = ['FedPUB', 'pfgnas-one', 'pfgnas']
        baselines = ['FedPUB', 'pfgnas-evo','pfgnas-random', 'pfgnas-one', 'pfgnas']
    elif max_res_line_dict or res_line_dict or accum_max_res_line_dict:
        baselines = ['gcn','FedPUB','fl-graphnas', 'pfgnas-evo',  'pfgnas-one', 'pfgnas']
        # baselines = ['FedPUB','fl-graphnas', 'pfgnas-evo', 'pfgnas-random', 'pfgnas-one', 'pfgnas']
    # elif res_line_dict:
    #     baselines = ['fl-graphnas','pfgnas-evo', 'pfgnas-random', 'pfgnas-one', 'pfgnas']
    else:
        baselines = gnn_types+fl_types+nas_types
    res_df = pd.DataFrame(index= baselines)
    person_res = {}
    for baseline in  baselines:
        # baseline = 'pfgnas'  ########### debug
        new_baseline_name = get_offcial_name(baseline)
        print(baseline)
        if baseline in gnn_types:
            if splitter == 'lda':
                directory = f'results_lda/FedAvg_{baseline}_{data_name}_{client}_{beta}'
            else:
                directory = f'results/{federate_method}_{baseline}_{data_name}_{client}'
        elif baseline in ['ditto','FedEM']:
            directory = f'results/{baseline}_gat_{data_name}_{client}'
        elif baseline in ['FedPUB']:
            if splitter == 'lda':
                directory = f'results_lda/{baseline}_unify_data_{data_name}_{client}_{beta}/200_0.01_100_4'
            else:
                directory = f'results/{baseline}_unify_data_{data_name}_{client}/200_0.01_100_4'
        elif baseline == 'darts':
            directory = f'results/{baseline}_fl-{baseline}_{data_name}_{client}'
        elif baseline in ['pfgnas-random', 'pfgnas-evo','pfgnas-one', 'pfgnas']:
            if splitter == 'lda':
                # directory = f'exp_lda/FedAvg_{baseline}_{data_name}_{beta}_{client}'
                directory = f'exp_lda/fedsageplus_{baseline}_{data_name}_{beta}_{client}'
                if baseline in ['pfgnas-random', 'pfgnas-evo', 'pfgnas']:
                    res_directory = f'results_lda/fedsageplus_pfgnas_{data_name}_{client}_{beta}'
                else:
                    # if beta ==10:
                    #     res_directory= f'results/FedAvg_{baseline}_{data_name}_{client}'
                    # else:
                    res_directory = f'results_lda/fedsageplus_{baseline}_{data_name}_{client}_{beta}'

            else:
                directory = f'exp/FedAvg_{baseline}_{data_name}_{client}'
                res_directory= f'results/FedAvg_{baseline}_{data_name}_{client}'

        elif baseline in ['fl-graphnas']:
            if splitter == 'lda':
                directory = f'exp_lda/FedAvg_{baseline}_{data_name}_{beta}_{client}'
            else:
                directory = f'exp/FedAvg_{baseline}_{data_name}_{client}'
            if res_line_dict:
                if splitter == 'lda':
                    res_path = f'results_lda/FedAvg_{baseline}_{data_name}_{client}_{beta}'
                else:

                    res_path = f'results/FedAvg_{baseline}_{data_name}_{client}'
        else:
            directory = f'results/FedAvg_{baseline}_{data_name}_{client}'
            res_directory = directory
        if not os.path.exists(directory):
            print(f"路径 '{directory}' 不存在")
            return
        if baseline in ['nas','pfgnas-one']:
            max_acc_key = None
            for root, dirs, files in os.walk(directory):
                for file in files:
                    if file == 'models_res_list.log':
                        file_path = os.path.join(root, file)
                        with open(file_path, 'r') as f:
                            data = f.readlines()
                        models_res_list = [json.loads(line.strip()) for line in data]
                        if res_line_dict or max_res_line_dict :
                            res_line_dict,max_res_line_dict,accum_max_res_line_dict = \
                                cal_round_avgAndmax_res_pfgnasone(models_res_list,res_line_dict,max_res_line_dict,accum_max_res_line_dict,new_baseline_name,fig_round)

                        # Merge and Remove Duplicates
                        merged_dict = {}
                        for item in models_res_list:
                            for key, value in item.items():
                                merged_dict.setdefault(key, {}).update(value)
                        max_acc_key = max(merged_dict, key=lambda k: float(merged_dict[k]["acc"].split("±")[0]))
                        if res_barplot_dict:
                            path = os.path.join(res_directory, f'{max_acc_key}_200_0.005_100_4')
                            if not os.path.exists(path):
                                print(f"The path '{path}' not exsit!")
                                return
                            original_res = get_original_res(path)
                            res_barplot_dict['beta'] += [beta] * len(original_res['acc'])
                            res_barplot_dict['Methods'] += [new_baseline_name] * len(original_res['acc'])
                            res_barplot_dict['Accuracy'] += original_res['acc']
                            res_barplot_dict['ROC_AUC'] += original_res['roc_auc']

                        max_acc_value = merged_dict[max_acc_key]
                        for key, value_string in max_acc_value.items():
                            # value, error = [f"{float(part) * 100:.2f}" for part in value_string.split("±")]
                            # res_df.loc[baseline, key] = f"{value}±{error}"
                            if isinstance(value_string, str) and "±" in value_string:
                                value, error = [f"{float(part) * 100:.2f}" for part in value_string.split("±")]
                                res_df.loc[baseline, key] = f"{value}±{error}"
                            else:
                                res_df.loc[baseline, key] = value_string
            # max_path = os.path.join(res_directory, f'{max_acc_key}_200_0.005_100_4')
            max_path = os.path.join(res_directory, f'{max_acc_key}_200_0.005_100_4')
            _, model_results = get_original_res(max_path)
            person_res[new_baseline_name],_ = calculate_average_dict_with_variance(model_results)
            # print(f'{baseline} max value in {max_acc_key}, the path is {max_path}')
            print(f'{baseline} max value in {max_acc_key}, the path is {max_path}')
        elif baseline == 'FedPUB':
            #res_barplot_dict = {'beta': [], 'acc': [], 'methods': []}
            all_avg_res = []
            for root, dirs, files in os.walk(directory):
                for dir_name in dirs[:]:
                    if dir_name == 'checkpt':
                        continue
                    dir_path = os.path.join(root, dir_name)
                    # extract results of clients
                    client_dict = {}
                    for i in range(client):
                        file_path = os.path.join(dir_path, f'client_{i}.txt')
                        client_dict[str(i+1)] = extract_client_fedpub(file_path)

                    file_path = os.path.join(dir_path,'global.txt')
                    last_values_dict = extract_client_fedpub(file_path)
                    client_dict['Global']  = last_values_dict
                    if res_barplot_dict:
                        all_avg_res.append(last_values_dict)
                    else:
                        all_avg_res.append(client_dict)
            person_res[new_baseline_name],_ = calculate_average_dict_with_variance(all_avg_res)
            for metric in person_res[new_baseline_name].keys():
                values = [float(value) for value in person_res[new_baseline_name][metric].values()]
                mean_value = round(np.mean(values),2)
                std_deviation = round(np.std(values),2)
                res_df.loc[baseline, metric] = str(mean_value) + "±" + str(std_deviation)    
            if res_barplot_dict:
                res_barplot_dict['beta']+= [beta]*len(all_avg_res)
                res_barplot_dict['Methods']+= [new_baseline_name] * len(all_avg_res)
                for metric in all_avg_res[0].keys():
                    values = [item[metric] for item in all_avg_res]
                    if metric == 'acc':
                        res_barplot_dict['Accuracy'] += values
                    elif metric == 'roc_auc':
                        res_barplot_dict['ROC_AUC']+= values
                    # mean_value = round(np.mean(values) *100,2)
                    # std_deviation = round(np.std(values) *100,2)
                    # res_df.loc[baseline, metric] = str(mean_value) + "±" + str(std_deviation)

        elif baseline == 'fl-graphnas':
            file_path = os.path.join(directory, 'res.log')
            if os.path.exists(file_path):
                with open(file_path, 'r') as f:
                    text = f.read()
                experiment_texts = re.findall(r'{.*?}', text)
                experiments = [ast.literal_eval(exp_text) for exp_text in experiment_texts]
                experiment = experiments[-1]
                for key, value_string in experiment.items():
                    if isinstance(value_string, str) and "±" in value_string:
                        value, error = [f"{float(part) * 100:.2f}" for part in value_string.split("±")]
                        res_df.loc[baseline, key] = f"{value}±{error}"
                    else:
                        res_df.loc[baseline, key] = value_string
            if res_line_dict:
                file_path = os.path.join(directory, 'response.log')
                if os.path.exists(file_path):
                    with open(file_path, 'r') as file:
                        text = file.read()
                result = [line.strip().replace('"', '') for line in text.split('\n')]
                result = [item for item in result if item != ""]
                current_max_acc = float('-inf')
                current_max_dict = {}
                if fig_round is None:
                    rounds = len(result)
                max_line = None
                for r in range(rounds):
                    line = result[r]

                    if data_name == 'pubmed':
                        original_res_path = os.path.join(res_path,f'{line}_200_0.001_100_4')
                    else:
                        original_res_path = os.path.join(res_path, f'{line}_50_0.001_100_4')
                    original_res,_= get_original_res(original_res_path )
                    res_line_dict = add_v(res_line_dict, original_res, r, new_baseline_name)
                    max_res_line_dict = add_v(max_res_line_dict, original_res, r, new_baseline_name)
                    # res_line_dict = cal_round_avg_res(models_res_list, res_line_dict, new_baseline_name)
                    if accum_max_res_line_dict:
                        mean_values = {key: sum(values) / len(values) for key, values in original_res.items()}
                        if mean_values['acc'] > current_max_acc:
                            current_max_acc = mean_values['acc']
                            current_max_dict = copy.deepcopy(original_res)
                            max_line = line
                        accum_max_res_line_dict = add_v(accum_max_res_line_dict, current_max_dict, r, new_baseline_name)

                max_path = os.path.join(res_path,f'{max_line}_200_0.001_100_4')
                _, model_results = get_original_res(max_path)
                person_res[new_baseline_name],_ = calculate_average_dict_with_variance(model_results)

                print(f'{baseline} max value in {max_line}, the path is {max_path}')

        elif baseline in ['pfgnas-evo','pfgnas-random','pfgnas']:
            max_acc_key = None
            # res_directory = None
            for root, dirs, files in os.walk(directory):
                for file in files:
                    if file == 'models_res_list.log':
                        file_path = os.path.join(root, file)
                        with open(file_path, 'r') as f:
                            data = f.readlines()
                        models_res_list = [json.loads(line.strip()) for line in data]
                        if res_line_dict and max_res_line_dict:
                            res_line_dict,max_res_line_dict,accum_max_res_line_dict = \
                                cal_round_avgAndmax_res(models_res_list,res_line_dict,max_res_line_dict,accum_max_res_line_dict,new_baseline_name,fig_round)

                        # Remove Duplicates of dict
                        merged_dict = {}
                        for item in models_res_list:
                            for key, value in item.items():
                                merged_dict.setdefault(key, {}).update(value)
                        max_acc_key = max(merged_dict, key=lambda k: float(merged_dict[k]['0']['test_acc'].split("±")[0])) # 'gdc3-ebh5-afi1'
                        if res_barplot_dict:
                            path = os.path.join(res_directory, f'{max_acc_key}_200_0.005_100_4')
                            if not os.path.exists(path):
                                print(f"The path '{path}' not exsit")
                                return
                            original_res,_ = get_original_res(path)
                            # res_barplot_dict = {'beta': [], 'acc': [], 'methods': [], 'roc_auc': []}
                            res_barplot_dict = add_v(res_barplot_dict, original_res, beta, new_baseline_name)
                            # res_barplot_dict['beta'] += [beta] * len(original_res['acc'])
                            # res_barplot_dict['Methods'] += [new_baseline_name] * len(original_res['acc'])
                            # res_barplot_dict['Accuracy'] += original_res['acc']
                            # res_barplot_dict['ROC_AUC'] += original_res['roc_auc']
                        max_acc_value = merged_dict[max_acc_key]['0']
                        for key, value_string in max_acc_value.items():
                            key = 'acc' if key =='test_acc' else 'roc_auc'
                            # value, error = [f"{float(part) * 100:.2f}" for part in value_string.split("±")]
                            # res_df.loc[baseline, key] = f"{value}±{error}"
                            # res_df.loc[baseline, key] = value
                            if isinstance(value_string, str) and "±" in value_string:
                                value, error = [f"{float(part) * 100:.2f}" for part in value_string.split("±")]
                                res_df.loc[baseline, key] = f"{value}±{error}"
                            else:
                                res_df.loc[baseline, key] = value_string
            # max_acc_key = 'fbd5-ice2-hfj2-hee5-baj1-hee5-jhf2-jce1-bfa2-fbd5'
            max_path = os.path.join(res_directory, f'{max_acc_key}_200_0.005_100_4')
            _, model_results = get_original_res(max_path)
            person_res[new_baseline_name],_ = calculate_average_dict_with_variance(model_results)
            print(f'{baseline} max value in {max_acc_key}, the path is {max_path}')
        elif baseline in gnn_types:
            max_path = os.path.join(directory, '200_0.005_100_4')
            _, model_results = get_original_res(max_path)
            person_res[new_baseline_name],res_avg_var_dict = calculate_average_dict_with_variance(model_results)
            res_df.loc[baseline, 'acc'] = res_avg_var_dict['acc']['Global']
            res_df.loc[baseline, 'roc_auc'] = res_avg_var_dict['roc_auc']['Global']
            print(f'{baseline} , the path is {max_path}')

        else:
            max_val = float('-inf')
            for root, dirs, files in os.walk(directory):
                for file in files:
                    if file == 'avg_res.log':
                        file_path = os.path.join(root, file)
                        with open(file_path, 'r') as f:
                            res_data = f.readlines()
                        last_line = res_data[-1].strip()
                        res_data = json.loads(last_line)

                        if float(res_data['acc'].split("±")[0]) > max_val:
                            max_val = float(res_data['acc'].split("±")[0])
                            for key, value_string in res_data.items():
                                # value, error = [f"{float(part) * 100:.2f}" for part in value_string.split("±")]
                                # res_df.loc[baseline, key] = f"{value}±{error}"
                                # res_df.loc[baseline, key] = value
                                if isinstance(value_string, str) and "±" in value_string:
                                    value, error = [f"{float(part) * 100:.2f}" for part in value_string.split("±")]
                                    res_df.loc[baseline, key] = f"{value}±{error}"
                                else:
                                    res_df.loc[baseline, key] = value_string
                        else:
                            continue


    print(res_df)
    res_df.to_csv(os.path.join(out_path,'res.log'), sep='\t', index_label='Type')
    return res_df,res_barplot_dict, res_line_dict,max_res_line_dict,accum_max_res_line_dict,person_res




def sort_dicts_by_test_metrics(merged_dict,reverse):
    sort_key = lambda x: (x[1]['0'].get('test_acc', 0), x[1]['0'].get('test_roc_auc', 0))
    sorted_items = sorted(merged_dict.items(), key=sort_key, reverse=reverse)
    sorted_dict = dict(sorted_items)
    return sorted_dict

def dict_to_string(input_dict):
    result_list = []

    for model, metrics in input_dict.items():
        result_item = f"Framework {model} achieves"

        for client, values in metrics.items():
            if client == '0':
                result_item += f" a test accuracy of {values['test_acc']} and a test ROC-AUC of {values['test_roc_auc']} on server."
            # else:
            #     result_string += f" a training accuracy of {values['train_acc']} and a training ROC-AUC of {values['train_roc_auc']} on client {client},"

        result_item = result_item.rstrip(",") + "\n"
        result_list.append(result_item)
        # result_string = result_string.rstrip(",")
    return result_list

def _dict_to_string(input_dict):
    result_string = ""

    for model, metrics in input_dict.items():
        result_string += f" Model {model} achieves a test accuracy of {metrics['acc']} and a test ROC-AUC of {metrics['roc_auc']} on server."
        result_string = result_string.rstrip(",") + "\n"

    return result_string


def get_client_server_results(path):
    model_results = []
    metric = None
    for root, dirs, files in os.walk(path):
        for dir_name in dirs[:]:
            if dir_name != "model":  # 
                dir_path = os.path.join(root, dir_name)
                best_res_log_path = os.path.join(dir_path, "eval_results_raw.log")
                if os.path.exists(best_res_log_path):
                    one_res_dict = extarct_client_server_res(best_res_log_path) #{'2': {'train_acc': 0.7204, 'train_roc_auc': 0.918}, '1': {'train_acc': 0.8225, 'train_roc_auc': 0.9561}, '3': {'train_acc': 0.4932, 'train_roc_auc': 0.9021}, '0': {'test_acc': 0.5622, 'test_roc_auc': 0.6593}}
                    model_results.append(one_res_dict)
    average_dict = get_avg_client_server_results(model_results)
    return average_dict

def get_avg_client_server_results(model_results):
    average_dict = {}

    for client_id, metrics_dict in model_results[0].items():
        if client_id != '0':
            test_acc_list = []
            test_roc_auc_list = []
            for results_dict in model_results:
                test_acc_list.append(results_dict[client_id]['test_acc'])
                test_roc_auc_list.append(results_dict[client_id]['test_roc_auc'])

            avg_test_acc = calculate_average_and_std(test_acc_list)
            avg_train_roc_auc = calculate_average_and_std(test_roc_auc_list)

            average_dict[client_id] = {'test_acc': avg_test_acc, 'train_roc_auc': avg_train_roc_auc}

    test_acc_list = []
    test_roc_auc_list = []
    for results_dict in model_results:
        test_acc_list.append(results_dict['0']['test_acc'])
        test_roc_auc_list.append(results_dict['0']['test_roc_auc'])

    avg_test_acc = calculate_average_and_std(test_acc_list)
    avg_test_roc_auc = calculate_average_and_std(test_roc_auc_list)

    average_dict['0'] = {'test_acc': avg_test_acc, 'test_roc_auc': avg_test_roc_auc}

    return average_dict

def calculate_average_and_std(data_list):
    values = [float(item) for item in data_list]
    avg_value = mean(values)
    std_value = stdev(values)
    return f"{avg_value:.4f}±{std_value:.4f}"






def cal_avg_client_res(path,upto=-1,eval_type='test'):
    # model_results = []
    # metric = None
    y_trials = []
    for root, dirs, files in os.walk(path):
        for dir_name in dirs[:]:
            dir_path = os.path.join(root, dir_name)
            clients = {}
            min_n_rnds = 9999
            y_trial={}
            for i, client in enumerate(glob.glob(os.path.join(dir_path, 'client*.txt'))):
                with open(client) as f:
                    client =json.loads(f.read())
                    n_rnds = len(client['log'][f'rnd_local_{eval_type}_acc'][:upto])
                    if n_rnds < min_n_rnds:
                        min_n_rnds = n_rnds
                    clients[i] = client['log'][f'rnd_local_{eval_type}_acc']
            y_trial[f'rnd_local_{eval_type}_acc'] = np.round(np.mean([client[:min_n_rnds] for i, client in clients.items()], 0) * 100, 2)
            y_trials.append(y_trial)
    return y_trials

def cal_round_avgAndmax_res_pfgnasone(models_res_list,res_line_dict,max_res_line_dict,accum_max_res_line_dict,method,fig_round=None):
    if fig_round is None:
        rounds = len(models_res_list)
    for r in range(rounds):
        # _, avg_dict = calculate_average_dict_simple(models_res_list[r])
        avg_dict = calculate_average_dict_pgnasone(models_res_list[r])
        res_line_dict = add_v(res_line_dict, avg_dict, r, method)

        # Average of Top5
        #
        # sorted_data = sorted(zip(avg_dict['acc'], avg_dict['roc_auc']), key=lambda x: (x[0], x[1]),
        #                      reverse=True)[:3]
        # avg_dict = {'acc': [item[0] for item in sorted_data], 'roc_auc': [item[1] for item in sorted_data]}

        sorted_data = sorted(zip(avg_dict['acc'], avg_dict['roc_auc']), key=lambda x: (x[0], x[1]), reverse=True)[:3]
        avg_dict = {'acc': [item[0] for item in sorted_data], 'roc_auc': [item[1] for item in sorted_data]}
        max_res_line_dict = add_v(max_res_line_dict, avg_dict, r, method)

        merged_dict = {key: value for d in models_res_list[:r+1] for key, value in d.items()}
        avg_dict = calculate_average_dict_pgnasone(merged_dict)
        sorted_data = sorted(zip(avg_dict['acc'], avg_dict['roc_auc']), key=lambda x: (x[0], x[1]), reverse=True)[:5]
        avg_dict = {'acc': [item[0] for item in sorted_data], 'roc_auc': [item[1] for item in sorted_data]}
        # avg_dict = {key: sorted(values, reverse=True)[:5] for key, values in avg_dict .items()}
        # _, avg_dict = calculate_average_dict_simple(result_dict)
        accum_max_res_line_dict = add_v(accum_max_res_line_dict, avg_dict, r, method)
        # accum_max_res_line_dict['Accuracy'] += avg_dict['acc']
        # accum_max_res_line_dict['ROC_AUC'] += avg_dict['roc_auc']
        # accum_max_res_line_dict['Round'] += [r] * len(avg_dict['acc'])
        # accum_max_res_line_dict['Methods'] += [method] * len(avg_dict['acc'])

    return res_line_dict, max_res_line_dict, accum_max_res_line_dict

def calculate_average_dict_pfgnasone():

    pass



if __name__ == '__main__':
    # res_dir = 'results/FedPUB_lr_cora_3/200_0.01_100_4/'
    # cal_avg_client_res(res_dir)
    #######################################################################################################################
    # splitter = 'lda'
    # betas = 0.2
    # data_names = ['pubmed']
    # clients = [3]
    # federate_method = 'fedsageplus'
    splitter = None
    betas = None
    data_names = ['cora']
    clients = [20]
    federate_method = 'FedAvg' 
    res_barplot_dict = None
    # res_line_dict = {'Round': [], 'Accuracy': [], 'Methods': [], 'ROC_AUC': []}
    # max_res_line_dict = {'Round': [], 'Accuracy': [], 'Methods': [], 'ROC_AUC': []}
    # accum_max_res_line_dict = {'Round': [], 'Accuracy': [], 'Methods': [], 'ROC_AUC': []}
    res_line_dict = None
    max_res_line_dict = None
    accum_max_res_line_dict = None
    fig_round = None
    for client in clients:
        for data_name in data_names:
            print(f'data:{data_name},client:{client}')
            res_df, _,res_line_dict,max_res_line_dict,accum_max_res_line_dict,person_res = extract_gnn_res(data_name, client,federate_method,splitter, betas,
                                                                        res_barplot_dict, res_line_dict,max_res_line_dict, accum_max_res_line_dict,
                                                                                                fig_round = fig_round)
            print('+++++++++++++++++++++++++++++++++++++++++++')
            # fig_path = f'res_fig/avg_perf_{data_name}_{client}.pdf'
            # max_fig_path = f'res_fig/max_avg_perf_{data_name}_{client}.pdf'
            # accum_max_fig_path = f'res_fig/accum_max_avg_perf_{data_name}_{client}.pdf'
            # res_line_df = pd.DataFrame(res_line_dict)

            #########################
            # Separate storage
            # plot_line_bond(res_line_df,x_name='Round',y_name='Accuracy', hua_name='Methods',fig_path=fig_path,order=None)
            #
            # plot_line_bond(max_res_line_dict, x_name='Round', y_name='Accuracy', hua_name='Methods', fig_path=max_fig_path,
            #                order=None)
            #
            # plot_line_bond(accum_max_res_line_dict, x_name='Round', y_name='Accuracy', hua_name='Methods',
            #                fig_path=accum_max_fig_path,
            #                order=None)
            #########################
            # merge two fig
            # fig, axes = plt.subplots(1, 2, figsize=(12, 6))
            # plt.sca(axes[0])
            # plot_line_bond(max_res_line_dict, x_name='Round', y_name='Accuracy', hua_name='Methods',
            #                fig_path=max_fig_path,
            #                order=None)
            # # plt.title("Average Accuracy /Round")
            # axes[0].set_title("Average Accuracy /Round", fontsize=20)
            #
            # plt.sca(axes[1])
            # plot_line_bond(accum_max_res_line_dict, x_name='Round', y_name='Accuracy', hua_name='Methods',
            #                               fig_path=accum_max_fig_path,
            #                               order=None)
            # # plt.title("Historical Best Accuracy /Round")
            # axes[1].set_title("Historical Best Accuracy /Round", fontsize=20)
            # plt.legend().set_visible(False)
            # # Adjust layout and show the figure
            # plt.tight_layout()
            # two_fig_path = f'res_fig/perf_{data_name}_{client}_{fig_round}.pdf'
            # plt.savefig(two_fig_path)
            # plt.show()
    ####################################################################################################################
            metric = 'roc_auc'
            # print(person_res)

            person_res_metric = {}
            for method, values in person_res.items():
                person_res_metric[method] = values[metric]
                #
                # for key, metrics in values.items():
                #     person_res_metric[method][key] = metrics

            print(person_res_metric)
            person_res_metric = pd.DataFrame(person_res_metric).reset_index()
            person_res_metric = person_res_metric.rename(columns={'index': 'Indicator'})

            # plot_radar_chart(person_res_metric)
            print(person_res_metric)
            person_res_metric.to_csv(f'results_lda/person_res_{metric}.csv', index=False)