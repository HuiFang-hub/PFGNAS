# from nas_bench_graph import light_read as lightread
# from nas_bench_graph import Arch
# from untils import best_link, give_bench, main_prompt, main_prompt_word, main_prompt_num
import json
import requests
import time
import numpy as np
#url = "https://api-hk.openai-sb.com/v1/chat/completions"
url = "https://api.openai-sb.com/v1/chat/completions"
headers = {
    "Content-Type": "application/json",
    "Authorization": "Bearer " + "sb-0538b4c5b057d61a6291b36877025b9150121dbff8c8be51"
}
gnn_list = [
    "gat",  # GAT with 2 heads 0
    "gcn",  # GCN 1
    "gin",  # GIN 2
    "cheb",  # chebnet 3
    "sage",  # sage 4
    "arma",     #   5
    "graph",  # k-GNN 6
    "fc",  # fully-connected 7
    "skip"  # skip connection 8
]

link_list = [
    [0, 0, 0, 0],
    [0, 0, 0, 1],
    [0, 0, 1, 1],
    [0, 0, 1, 2],
    [0, 0, 1, 3],
    [0, 1, 1, 1],
    [0, 1, 1, 2],
    [0, 1, 2, 2],
    [0, 1, 2, 3]
]

operation_dict ={'GCN':'gcn', 'GAT':'gat', 'GraphSAGE':'sage', 'GIN':'gin', 'ChebNet':'cheb', 'ARMA':'arma', 'k-GNN':'graph', 'skip':'skip', 'fully-connected-layer':'fc'}

dataname_list = 'cora'.split()# citeseer pubmed arxiv

system_content = '''Please pay special attention to my use of special markup symbols in the content below.The special markup symbols is # # ,and the content that needs special attention will be between #.'''

for dataname in dataname_list:
    link = best_link(dataname)
    link_str = ''.join(str(i) for i in link)
    bench = give_bench(dataname,link)

    filename = 'history/'+dataname+'/'+link_str
    filename_message = filename + 'messages_num2.json'
    filename_performance = filename + 'performance_num2.json'

    messages = [
        {"role": "system", "content": system_content},
        {"role": "user", "content": main_prompt_num(link=tuple(link), dataname=dataname, stage=0)},
    ]

    arch_list = []
    acc_list = []
    performance_history = []
    messages_history = []
    iterations=15
    for iteration in range(iterations):
        da = {
            "model": "gpt-4",#"gpt-3.5-turbo"-0314
            "messages": messages,
            "temperature": 0
        }

        response = requests.post(url, headers=headers, data=json.dumps(da))
        res = response.json()


        messages.append(res)
        messages_history.append(messages)

        with open(filename_message, 'w') as f:
            json.dump(messages_history, f)

        res_temp = res['choices'][0]['message']['content']
        #print(res_temp)
        input_lst = res_temp.split('model:')
        for i in range(1, len(input_lst)):
            operations_str = input_lst[i].split('[')[1].split(']')[0]
            operations_list = operations_str.split(',')
            operations_list_str = [a.replace(" ", "") for a in operations_list]
            #operations = [operation_dict[op] for op in operations_list_str]
            if operations_list_str==['8', '8', '8', '8']:
                continue
            operations = [gnn_list[int(op)] for op in operations_list_str]
            arch = Arch(link, operations)
            info = bench[arch.valid_hash()]
            arch_list.append({'arch_Operations': operations_str})
            acc_list.append(info['perf'])

            performance = {
                'arch_Struct': arch.link,
                'arch_Operations': arch.ops,
                'perf': info['perf'],
                'bench': info
            }
            print(iteration+1, operations_str,'test:',info['perf'],'rank',info['rank'])

            performance_history.append(performance)

        with open(filename_performance, 'w') as f:
            json.dump(performance_history, f)
        messages = [
            {"role": "system", "content": system_content},
            {"role": "user", "content": main_prompt_num(link=tuple(link), dataname=dataname, arch_list=arch_list, acc_list=acc_list, stage=iteration+1)},
        ]
print(1)