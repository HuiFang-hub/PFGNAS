import pandas as pd
import os
import json
import sys
current_dir = os.path.dirname(os.path.abspath(__file__))
root_path = os.path.dirname(current_dir)
sys.path.append(root_path)

from lib.results_lib import *
import numpy as np
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



def res_different_llms(folder_p,subfolder_prefix,llms):
    res_df = pd.DataFrame(index= llms)
    for llm in llms:
        # find res
        subfolder_p = subfolder_prefix+llm
        directory = os.path.join(folder_p,subfolder_p)
        file_path = find_file('models_res_list.log', directory)
        with open(file_path, 'r') as f:
            data = f.readlines()
        models_res_list = [json.loads(line.strip()) for line in data] 
        
        # Merge and Remove Duplicates 
        merged_dict = {}
        for item in models_res_list:
            for key, value in item.items():
                merged_dict.setdefault(key, {}).update(value)
        
        # extract best res according to 'test_acc'       
        max_acc_key = max(merged_dict, key=lambda k: float(merged_dict[k]['0']['test_acc'].split("±")[0])) # 'gdc3-ebh5-afi1'           
        max_acc_value = merged_dict[max_acc_key]['0']  
         
        # get results of different metric of the best res
        res_df =  get_best_res(max_acc_value,res_df,llm)
        res_df.loc[llm, 'model'] = max_acc_key
    return res_df

def res_client_num(data_name, client, gnn_types,fl_types,nas_types,federate_method='FedAvg'):
    baselines = gnn_types+fl_types+nas_types
    res_df = pd.DataFrame()
    for baseline in  baselines:
        new_baseline_name = get_offcial_name(baseline)
        directory = get_res_path(baseline, gnn_types,data_name, client,federate_method='FedAvg')
        if baseline in ['nas','pfgnas-one']:
            merged_dict = get_merged_dict(directory)
                    
            # extract best res according to 'test_acc'  
            max_acc_key = max(merged_dict, key=lambda k: float(merged_dict[k]["acc"].split("±")[0]))            
            max_acc_value = merged_dict[max_acc_key]

            # extract best res according to 'test_acc'      
            res_df =  get_best_res(max_acc_value,res_df,new_baseline_name)
        elif baseline == 'FedPUB':
            all_avg_res = []
            for root, dirs, files in os.walk(directory):
                for dir_name in dirs:
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
                    all_avg_res.append(client_dict)
            res_avg_dict,res_avg_var_dict = calculate_average_dict_with_variance(all_avg_res)
            for metric in res_avg_dict.keys():
                values = [float(value) for value in res_avg_dict[metric].values()]
                mean_value = round(np.mean(values),2)
                std_deviation = round(np.std(values),2)
                metric = unify_metric(metric)
                res_df.loc[new_baseline_name, metric] = str(mean_value) + "±" + str(std_deviation)

        elif baseline in gnn_types:
            path = find_file('avg_res.log', directory)
            with open(path, 'r') as f:
                last_line = f.readlines()[-1].strip()
            res_data = json.loads(last_line)
            
            # max_acc_value = float(res_data['acc'].split("±")[0])
            res_df =  get_best_res(res_data,res_df,new_baseline_name)
           
        elif baseline in ['pfgnas-evo','pfgnas-random','pfgnas']:
            file_path = find_file('models_res_list.log', directory)
            with open(file_path, 'r') as f:
                data = f.readlines()
            models_res_list = [json.loads(line.strip()) for line in data] 
            
            # Merge and Remove Duplicates 
            merged_dict = {}
            for item in models_res_list:
                for key, value in item.items():
                    merged_dict.setdefault(key, {}).update(value)
            
            # extract best res according to 'test_acc'       
            max_acc_key = max(merged_dict, key=lambda k: float(merged_dict[k]['0']['test_acc'].split("±")[0])) # 'gdc3-ebh5-afi1'           
            max_acc_value = merged_dict[max_acc_key]['0']  
            
            # get results of different metric of the best res
            res_df = get_best_res(max_acc_value,res_df,new_baseline_name)
        elif baseline == 'fl-agnns':
            max_val = float('-inf')
            path = find_file('avg_res.log', directory)
            with open(path, 'r') as f:
                last_line = f.readlines()[-1].strip()
            res_data = json.loads(last_line)
            if float(res_data['acc'].split("±")[0]) > max_val:
                max_val = float(res_data['acc'].split("±")[0])
                # for key, value_string in res_data.items():
                #     # value, error = [f"{float(part) * 100:.2f}" for part in value_string.split("±")]
                #     # res_df.loc[baseline, key] = f"{value}±{error}"
                #     # res_df.loc[baseline, key] = value
                #     if isinstance(value_string, str) and "±" in value_string:
                #         value, error = [f"{float(part) * 100:.2f}" for part in value_string.split("±")]
                #         res_df.loc[baseline, key] = f"{value}±{error}"
                #     else:
                #         res_df.loc[baseline, key] = value_string
            # max_acc_value = float(res_data['acc'].split("±")[0])
                res_df =  get_best_res(res_data,res_df,new_baseline_name)
            
    
    return res_df








if __name__ == '__main__':
    # task
    diffirent_llm = False
    diffirent_client_num = False
    prompt_ablation = True
    
    # folder path
    if diffirent_llm:
        folder_p = 'exp_llm'
        subfolder_prefix = 'FedAvg_pfgnas_cora_3_'
        llms = ['glm','palm','gpt']
        res_df = res_different_llms(folder_p,subfolder_prefix,llms)
        print(res_df)
        
    if diffirent_client_num:
        data_name = 'citeseer'
        client = 20
        # methods
        gnn_types = ['gcn', 'sage', 'gat', 'gin', 'gpr', 'sgc', 'arma',  'appnp']
        fl_types=['FedPUB'] 
        nas_types = ['fl-agnns', 'pfgnas']
        res_df = res_client_num(data_name, client, gnn_types,fl_types,nas_types)
        print(res_df)
        
    if prompt_ablation:
        folder_p = 'exp_prompt_ab'
        subfolder_prefix = 'FedAvg_pfgnas_pubmed_10_'
        prompt = ['1','2','3','0']
        res_df = res_different_llms(folder_p,subfolder_prefix,prompt)
        print(res_df)
        
        
        
