# -*- coding: utf-8 -*-
# @Time    : 2023/12/12 20:32
# @Function:
import random
import string
import re
def generate_random_string(length):
    """生成随机字符串，范围在[a, k]之间"""
    return ''.join(random.choice(string.ascii_lowercase[0:11]) for _ in range(length))

def generate_random_string_(length):
    """生成随机字符串，范围在[a, i]之间"""
    return ''.join(random.choice(string.ascii_lowercase[0:5]) for _ in range(length))


def generate_random_num(length):
    """生成随机字符串，范围在[1,9]之间"""
    return ''.join(str(random.randint(1, 9)) for _ in range(length))


def generate_random_num_(length):
    """生成随机字符串，范围在[1,9]之间"""
    return ''.join(str(random.randint(1, 5)) for _ in range(length))

def generate_model_combination(n,operation_num):
    """生成一组GNN模型的组合"""
    return '-'.join([generate_random_string(operation_num-1) + generate_random_num_(1) for _ in range(n)])
    # return '-'.join(['\''+ generate_random_string(operation_num-1) + generate_random_num_(1)+'\'' for _ in range(n)])
    # return [generate_random_string(operation_num-1) + generate_random_num_(1) for _ in range(n)]


def generate_federated_models_code(n, num_models,operation_num=4):
    """生成多组联邦模型"""
    federated_models = []
    for i in range(1, num_models + 1):
        clients = '-'.join([f'client{j}' for j in range(1, n + 1)])
        model_combination = generate_model_combination(n,operation_num)
        # random_num = str(random.randint(1, 5))
        # federated_models.append(f"{i}. {clients}: {model_combination}")
        federated_models.append(f"{i}: {model_combination}")
    return federated_models


def generate_federated_models(n,num_models):
    gnn_operations = ['gcn', 'sage', 'gpr', 'gat', 'gin', 'fc', 'sgc', 'arma', 'appnp', 'identity', 'zero']
    activation_functions = ['sigmoid', 'tanh', 'relu', 'linear', 'elu']


    federated_models = []
    for i in range(1, num_models + 1):
        model_list = []
        for _ in range(n):
            gnn_op1 = random.choice(gnn_operations)
            gnn_op2 = random.choice(gnn_operations)
            gnn_op3 = random.choice(gnn_operations)
            activation_op = random.choice(activation_functions)
            sub_model = [gnn_op1, gnn_op2, gnn_op3, activation_op]
            model_list.append(sub_model)
        federated_models.append(f"{i}: {model_list}")
        # print(f"{i}: {model_list}")
    return federated_models

# def generate_random_string(length):
#     """生成随机字符串，范围在[a, i]之间"""
#     return ''.join(random.choice('abcdefghi') for _ in range(length))

def generate_model_string(n):
    """生成一个包含3个字母和1个数字的字符串"""
    letters = generate_random_string(n-1)
    number = str(random.randint(1, 5))
    return f"{letters}{number}"

def random_generate_model_lists(n,ge_n):
    """生成包含n个字符串的列表"""
    random.seed()
    model_lists = []
    for _ in range(ge_n):
        model_string = '-'.join([generate_model_string(4) for _ in range(n)])
        model_lists.append(model_string)
    return model_lists

def random_global_model_list(num_models):
    gnn_operations = ['gcn', 'sage', 'gpr', 'gat', 'gin', 'fc', 'sgc', 'arma', 'appnp', 'identity', 'zero']
    activation_functions = ['sigmoid', 'tanh', 'relu', 'linear', 'elu']
    federated_models = []
    federated_models = []
    for i in range(1, num_models + 1):
        gnn_op1 = random.choice(gnn_operations)
        gnn_op2 = random.choice(gnn_operations)
        gnn_op3 = random.choice(gnn_operations)
        activation_op = random.choice(activation_functions)
        sub_model = [gnn_op1, gnn_op2, gnn_op3, activation_op]
        federated_models.append(f"{i}: {sub_model}")
        # print(f"{i}: {model_list}")
    return federated_models


if __name__ == '__main__':
    n = 5
    num_models=10
    # model_choose_list = generate_federated_models(n, num_models)
    model_choose_list =random_global_model_list(num_models)
    print(model_choose_list)
    # result = result[0].split(':')[-1]
    # print( model_choose_list)
    # model_example = ''' For a federated model with {} clients, your response only need include the federated models list\n{}'''.format(
    #     num_models,
    #     ''.join(['{}\n'.format(model) for model in result[:3]]))
    # print(  model_example )
    # pattern = re.compile(r':\s*([\w-]+)')
    # model_lists = re.findall(pattern, model_example )
    # print(model_lists)
    # 使用正则表达式匹配并提取子字符串
    # 定义正则表达式模式
    # pattern = re.compile(r"'(.*?)'")
    #
    # # 提取目标字符串并组成新的列表
    # result_list = [pattern.findall(item) for item in result]
    #
    # # 将列表中的元素连接为字符串
    # final_result = ['-'.join(sublist) for sublist in result_list]

    # print(final_result)
    # model_example = ''' Assuming there are {} clients in each federated scenario, please provide 5 different personalized federated models. For example:\n{}'''.format(
    #     5, 5,
    #     ''.join(['{}\n'.format(model) for model in result[:2]]))
    # pattern = r'\d+\.\s(\[\[.*?\]\])'
    # matches = re.findall(pattern,  model_example, re.DOTALL)
    # model_lists = [eval(match) for match in matches]
    # print(model_lists)
    # mapping_dict = {'gcn': 'a', 'sage': 'b', 'gpr': 'c', 'gat': 'd', 'gin': 'e',
    #                 'fc': 'f', 'sgc': 'g', 'arma': 'h', 'appnp': 'i', 'identity': 'j',
    #                 'zero': 'k', 'sigmoid': '1', 'tanh': '2', 'relu': '3', 'linear': '4',
    #                 'elu': '5'}
    ################################
    # model_example = ''' Assuming there are {} clients in each federated scenario, please provide 10 different personalized federated framework. For example:\n{}'''.format(5,
    #              ''.join(['{}\n'.format(model) for model in model_choose_list[:2]]))
    # model_example2 = '...\n'
    # model_example3 = ''.join(['{}\n'.format(model) for model in model_choose_list[-1:]])
    # print(model_example+model_example2+model_example3)
    ####################################
    # print(random_global_model_list(4,10))
    ####################################
