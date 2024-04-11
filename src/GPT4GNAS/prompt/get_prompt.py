# -*- coding: utf-8 -*-
# @Time    : 2023/10/19 10:17
# @Function:
from lib.extarct_final_result import extarct_metric, extarct_all,pf_extarct_all,sort_dicts_by_test_metrics,dict_to_string,_dict_to_string
from lib.result_resolver import calculate_average_dict
from src.NAS.module_builder import get_model_name
from src.GPT4GNAS.prompt.get_model_list import generate_federated_models, random_global_model_list
import json
def prefix_prompt():
    system_content = '''Please pay special attention to my use of special markup symbols in the content below. The special markup symbol is #, and the content that needs special attention will be between #.'''
    return system_content

def get_search_space(struct_dict,link, comprehensive = True):
    if comprehensive:
        Search_Space = '''And the objective is to maximize the accuracy and ROC-AUC value of model.
           A GNN architecture is defined as follows: 
           {
               The first operation is input, the last operation is output, and the intermediate operations are candidate operations.
               The adjacency matrix  of operation connections is as follows: ''' + str(struct_dict[link]) + '''where the (i,j)-th element in the adjacency matrix denotes that the output of operation $i$ will be used as  the input of operation $j$.
           }
           There are 9 most widely adopted GNN operations: gcn, sage, gpr, gat, gin, fc, sgc, arma and appnp. 
           For convenience, we use alphabetical codes to represent these GNN operations: {'1':'gcn', '2':'sage', '3':'gpr', '4':'gat','5': 'gin', '6':'fc', '7':'sgc', '8':'arma', '9':'appnp'}
           The definition of '1' GNN operation (i.e., gcn) is as follows:
           {
               The graph convolutional operation from the "Semi-supervised Classification with Graph Convolutional Networks" paper.
               Its node-wise formulation is given by:
               $$\mathbf{x}_i^{\prime}=\\boldsymbol{\Theta}^{\\top} \sum_{j \in \mathcal{N}(i) \cup\{i\}} \\frac{e_{j, i}}{\sqrt{\hat{d}_j \hat{d}_i}} \mathbf{x}_j$$
               with$\hat{d}_i=1+\sum_{j \in \mathcal{N}(i)} e_{j, i}$, where $e_{j, i}$ denotes the edge weight from source node $j$ to target node i (default: 1.0)
           }
           The definition of '2' GNN operation (i.e., sage) is as follows:
           {
               The GraphSAGE operation from the "Inductive Representation Learning on Large Graphs" paper
               $$\mathbf{x}_i^{\prime}=\Theta_1 \mathbf{x}_i+\Theta_2 \cdot {mean}_{j \in \mathcal{N}(i)} \mathbf{x}_j$$
           }
           The definition of '3' GNN operation (i.e., gpr) is as follows:
           {
               The Generalized PageRank operation from the "Adaptive Universal Generalized PageRank Graph Neural Network" paper.
               Generalized PageRank (GPR) GNN architecture adaptively learns the GPR weights so as to jointly optimize node feature and topological information extraction, regardless of the extent to which the node labels are homophilic or heterophilic.
               Its formulation is given by:
               $$\mathbf{H}_{\mathrm{GCN}}^{(k)}=\operatorname{ReLU}\left(\tilde{\mathbf{A}}_{\mathrm{sym}} \mathbf{H}_{\mathrm{GCN}}^{(k-1)} \mathbf{W}^{(k)}\right), \hat{\mathbf{P}}_{\mathrm{GCN}}=\operatorname{softmax}\left(\tilde{\mathbf{A}}_{\mathrm{sym}} \mathbf{H}_{\mathrm{GCN}}^{(K-1)} \mathbf{W}^{(k)}\right)$$    }
               where 
               $$\hat{\mathbf{P}}=\operatorname{softmax}(\mathbf{Z}), \mathbf{Z}=\sum_{k=0}^K \gamma_k \mathbf{H}^{(k)}, \mathbf{H}^{(k)}=\tilde{\mathbf{A}}_{\mathrm{sym}} \mathbf{H}^{(k-1)}, \mathbf{H}_{i:}^{(0)}=f_\theta\left(\mathbf{X}_{i:}\right)$$
           The definition of '4' GNN operation (i.e., gat) is as follows:
           {
               The graph attentional operation from the "Graph Attention Networks" paper.
               Its formulation is given by:
               $$\mathbf{x}_i^{\prime}=\\alpha_{i, i} \Theta \mathbf{x}_i+\sum_{j \in \mathcal{N}(i)} \\alpha_{i, j} \Theta \mathbf{x}_j$$
               where the attention coefficients $\\alpha_{i, j}$ are computed as
               $$\\alpha_{i, j}= \\frac{\exp \left({LeakyReLU}\left(\mathbf{a}^{\\top}\left[\\boldsymbol{\Theta} \mathbf{x}_i | \Theta \mathbf{x}_j\\right]\\right)\\right)}{\sum_{k \in \mathcal{N}(i) \cup\{i\}} \exp \left({LeakyReLU}\left(\mathbf{a}^{\\top}\left[\Theta \mathbf{x}_i | \Theta \mathbf{x}_k\\right]\\right)\\right)} $$
           }

           The definition of '5' GNN operation (i.e., gin) is as follows:
           {
               The graph isomorphism operation from the "How Powerful are Graph Neural Networks?" paper.
               Its formulation is given by:
               $$\mathbf{x}_i^{\prime}=h_{\Theta}\left((1+\epsilon) \cdot \mathbf{x}_i+\sum_{j \in \mathcal{N}(i)} \mathbf{x}_j\\right)$$
               here $h_{\Theta}$ denotes a neural network, i.e. an MLP.
           }

           The definition of the '6' GNN operation (i.e., fc) is as follows:
           {
               $$\mathbf{x}^{\prime}=f(\Theta x+b)$$
           }
           The definition of the '7' GNN operation (i.e., sgc) is as follows:
           {
               The Simple Graph Convolution (SGC) operation from the "Simplifying Graph Convolutional Networks" paper.
               It reduces the excess complexity of GCN through successively removing nonlinearities and collapsing weight matrices between consecutive layers.        
           }

           The definition of '8' GNN operation (i.e., arma) is as follows:
           {
               The ARMA graph convolutional operation from the "Graph Neural Networks with Convolutional ARMAFilters" paper.
               It is a graph neural network implementation of the auto-regressive moving average (ARMA) filter with a recursive and distributed formulation, obtaining a convolutional layer that is efficient to train, localized in the node space, and can be transferred to new graphs at test time.
               Its formulation is given by:
               $$\mathbf{X}^{\prime}= \mathbf{X}_1^{(1)}$$
               with $\mathbf{X}_1^{(1)}$ being recursively defined by
               $\mathbf{X}_1^{(1)}=\sigma\left(\hat{\mathbf{L}} \mathbf{X}_1^{(0)} \Theta+\mathbf{X}^{(0)} \mathbf{V}\\right)$
               where $\hat{\mathbf{L}}=\mathbf{I}-\mathbf{L}=\mathbf{D}^{-1 / 2} \mathbf{A} \mathbf{D}^{-1 / 2}$ denotes the modified Laplacian $\mathbf{L}=\mathbf{I}-\mathbf{D}^{-1 / 2} \mathbf{A} \mathbf{D}^{-1 / 2}$
           }
           The definition of '9' GNN operation (i.e., appnp) is as follows:
           {
               The personalized propagation of neural predictions (PPNP) operation from the "Graph Neural Networks with Convolutional ARMAFilters" paper, APPNP is the fast approximation of PPNP.
               The model’s training time is on par or faster and its number of parameters on par or lower than previous models. It leverages a large, adjustable neighborhood for classification and can be easily combined with any neural network.
               Its formulation is given by:
               $$ \boldsymbol{Z}_{\mathrm{PPNP}}=\operatorname{softmax}\left(\alpha\left(\boldsymbol{I}_n-(1-\alpha) \hat{\tilde{\boldsymbol{A}}}\right)^{-1} \boldsymbol{H}\right), \quad \boldsymbol{H}_{i,:}=f_\theta\left(\boldsymbol{X}_{i,:}\right)$$
               where $\hat{\tilde{A}}=\tilde{D}^{-1 / 2} \tilde{\boldsymbol{A}} \tilde{D}^{-1 / 2}$ denotes is the symmetrically normalized adjacency matrix with self-loops.
           }
           Besides, the activation function introduces non-linearity to the network, allowing it to learn from complex patterns and relationships in the data. It takes the weighted sum of inputs and bias from the last layer of the model, applies a transformation, and produces the final output of the model. 
           There are 5 most widely adopted activation function operations: sigmoid,tanh,relu,linear,elu,
           For convenience, we use number to represent these activation function operations: {'a':'sigmoid','b':'tanh', 'c':'relu', 'd':'linear','5': 'elu'}
           The definition of 'a' activation function operation (i.e., sigmoid) is as follows:
           {
           The output range of the Sigmoid function is (0, 1). Its formulation is given by:
           $$f(x)=\frac{1}{1+e^{-x}}$$
           }
           The definition of 'b' activation function operation (i.e., tanh) is as follows:
           {
           The output range of the hyperbolic tangent activation functionis (-1, 1). Its formulation is given by:
           $$f(x)=\tanh (x)=\frac{e^x-e^{-x}}{e^x+e^{-x}}=\frac{2}{1+e^{-2 x}}-1$$
           }
           The definition of 'c' activation function operation (i.e., relu) is as follows:
           {
           The ReLU activation function is a piecewise linear function. Its formulation is given by:
           $$f(x)=max(0,1)$$
           }
           The definition of 'd' activation function operation (i.e., linear) is as follows:
           {
           The linear activation function is a simple identity function that returns the input value intact. Its formulation is given by:
            $$f(x)=x$$
           }
           The definition of 'e' activation function operation (i.e., elu) is as follows:
           {
           The ELU activation function outputs cases less than zero in a similar way to exponential calculations. Its formulation is given by:
          $$\mathrm{g}(x)=\mathrm{ELU}(x)=\left\{\begin{array}{r}
           x, x>0 \\
           \alpha\left(\mathrm{e}^x-1\right), x \leqslant 0
           \end{array}\right.
           $$
           }
           '''
    else:
        Search_Space =  '''This can be achieved through a personalized federated learning framework, which is defined as follows: 
           {It consists of sub-architectures from n clients. For each sub-architecture, the initial operation is the input, the final operation is the output, and the intermediate 4 operators consisit of candidate GNN operations and activation function operators of search sapce. }
           In search space, there are 9 most widely adopted GNN operators: 'gcn', 'sage', 'gpr', 'gat', 'gin', 'fc', 'sgc', 'arma', 'appnp', 'identity' and 'zero', you should consider the advantages and disadvantages of these GNN operators.
           Besides, the activation function introduces non-linearity to the network, allowing it to learn from complex patterns and relationships in the data. It takes the weighted sum of inputs and bias from the last layer of the model, applies a transformation, and produces the final output of the model. 
           There are 5 most widely adopted activation function operators: sigmoid, tanh, relu, linear and elu.
           '''

    return Search_Space


def main_prompt_word(cfg,struct_dict,link, models_res_list=None,stage=0):
    if cfg.model.type == 'pfgnas':
        Search_Task= f'A federated scenario refers to a situation where there are multiple clients (for example, multiple organizations, devices, or users) working together to achieve a goal, but there may be a need for data isolation and privacy protection between them. ' \
                     f'In a federated graph scenario, with {cfg.federate.client_num} clients, each holding a segment of the {cfg.data.type} graph dataset, the task is to customize the choice of the best sub-model for each client within the search space. This is done with the goal of maximizing accuracy and ROC-AUC values across all clients. '
    else:
        Search_Task =f'The task is to choose the best GNN architecture and the best activation function on a given dataset. The model will be trained and tested on {cfg.data.type}. '

    Search_Space = get_search_space(struct_dict,link, comprehensive = False)

    if cfg.model.type == 'pfgnas':
        Search_Strategy = '''Once again, your task is to search the optimal personalized federated models, this is, help all client search the optimal intermediate operations of sub-models on a given experimental dataset. 
        The search strategy we set encompasses two stages: 
        1. Exploration Stage: in the initial stage of the search, you should focus more on exploring the entire search space randomly, rather than just focusing on the current local optimal results. 
        2. Exploitation Stage: with a certain amount of experimental results, in the exploitation stage, you can iteratively refine the search for the best operation lists by querying new lists based on existing ones and their corresponding performance. You should employ optimization algorithms such as Bayesian optimization or evolutionary algorithms to more effectively explore the optimal combinations of operations instead of random selection.'''
    else:
        Search_Strategy = '''Once again, your task is to help me find the optimal combination of operations while specifying experimental dataset. The main difficulty of this task is how to reasonably arrange the selection strategy of the operation list, and each selected operation list corresponds to the highest accuracy that the operation can achieve. You should select a new operation list to query based on the existing operation lists and their corresponding accuracy, in order to iteratively find the best operation list.

            At the beginning, when there were few experimental results, we in the Exploration phase, you need to explore the operation space and identify which operation lists are promising. You can #randomly# select a batch of operation lists corresponding to each layer and evaluate their performance. Afterwards, we can sort the operation lists based on their accuracy and select some well performing operation lists as candidates for our Exploitation phase.
            
            When we have a certain amount of experimental results, we are in the Exploitation phase, you need focus on improving search by exploring the operating space more effectively. You can use optimization algorithms, such as Bayesian optimization or Evolutionary algorithm, to search for the best combination of operations, rather than randomly selecting the list of operations.
            '''
    # Part 1: explain  GNAS
    user_input = Search_Task + Search_Space + Search_Strategy

    # Exploration phase
    notice1 = '''\n# Now, it should be in the Exploration stage. You can randomly select from operators, considering some well-performing lists as candidates for the exploitation stage.#\n\n'''
    # Exploitation phase
    notice2 = '''\n#Due to the availability of a certain number of experimental results, it is currently in the Exploitation stage. You should choose nearby samples that temporarily display the best results for searching, especially those that rank in the top 10% or 20% of existing experimental results. At the same time, you should try to avoid exploring sample structures with poor results, which can cause waste.#\n\n'''
    if cfg.model.type == 'pfgnas':
        model_choose_list = generate_federated_models(cfg.federate.client_num, num_models=5)

        # suffix = ''' Please do not include anything other than the model list in your response. For simplicity, please use the following #9 letters# in the alphabet to represent GNN operations:
        # #'a':'gcn', 'b':'sage', 'c':'gpr', 'd':'gat', 'e': 'gin', 'f':'fc', 'g':'sgc', 'h':'arma', 'i':'appnp','j':'identity', 'k':'zero'# ,
        # and represent activation function operations with these #5 numbers#:
        # #'1':'sigmoid','2':'tanh', '3':'relu', '4':'linear', '5':'elu' #.
        # Each sub-model for each client should include #4 operations# : the first three operations are GNN operations, and the fourth is an activation function operation, such as 'aaa2'.
        # And each federated model is composed of sub-models from #{} clients#, for example, a federated model can be represented as a list: {}. In your response, adhere strictly to the format exemplified below, for example:'''.format(cfg.federate.client_num, model_choose_list[0].split(':')[-1])

        suffix = ''' there are 11 GNN operators: ['gcn', 'sage', 'gpr', 'gat',  'gin', 'fc', 'sgc', 'arma', 'appnp', 'identity', 'zero'], and there are 5 activation function operators: ['sigmoid', 'tanh', 'relu', 'linear', 'elu' ]. 
               Each sub-architecture for each client should include #4 operators# : the first three operators are GNN operations, and the fourth is an activation function operator, such as ['fc', 'fc', 'gpr', 'sigmoid']. 
               And each federated framework is composed of #{} sub-architectures#, for example, a federated framework can be represented as a list: {}. In your response, adhere strictly to the format exemplified below, for example:'''.format(
            cfg.federate.client_num, model_choose_list[0].split(':')[-1])

        model_example = ''' Assuming there are {} clients in each federated scenario, please provide #5# different personalized federated framework. For example:\n{}'''.format(cfg.federate.client_num,
             ''.join(['{}\n'.format(model) for model in model_choose_list[:2]]))
        model_example2 = '...\n'
        model_example3 = ''.join(['{}\n'.format(model) for model in model_choose_list[-1:]])
        model_example = model_example + model_example2 + model_example3
        suffix += model_example

        #model_choose_list = '1. client1-client2-client3: bfg3-ccf4-dac1',
        # '2. client1-client2-client3: ifh5-aci1-egi2',
        # '3. ...
        #  ...
        # '10. client1-client2-client3: ahh4-caf3-hfe5'

    else:
        model_choose_list =random_global_model_list(num_models=10)
        # suffix = '''Please do not include anything other than the operation list in your response.
        #     And you should give 10 different models at a time, one model contains #4# operations. The first three are chosen from GNN operations (i.e., 'a':'gcn', 'b':'sage', 'c':'gpr', 'd':'gat','e': 'gin', 'f':'fc', 'g':'sgc', 'h':'arma', 'i':'appnp','j':'identity', 'k':'zero'), and the last one is chosen from activation function operation('1':'sigmoid','2':'tanh', '3':'relu', '4':'linear','5': 'elu'), chosen from the search space described above. Please do not give repeated models.
        #     Please represent the selected operations using those key(i.e.,letters and numbers). Your response only need include the operation list, for example:
        #     1: edb2
        #     2: cci1
        #     3...
        #     ......
        #     10: fha5.
        #     And The response you give must strictly follow the format of this example. '''
        suffix = '''there are 11 GNN operators: ['gcn', 'sage', 'gpr', 'gat',  'gin', 'fc', 'sgc', 'arma', 'appnp', 'identity', 'zero'], and there are 5 activation function operators: ['sigmoid', 'tanh', 'relu', 'linear', 'elu' ]. 
         Each architecture should include #4 operators# : the first three operators are GNN operations, and the fourth is an activation function operator, such as {}. In your response, adhere strictly to the format exemplified below, for example:'''\
            .format(model_choose_list[5].split(':')[-1])
        model_example = ''' Please provide #10# different personalized federated framework. For example:\n{}'''.format(
            ''.join(['{}\n'.format(model) for model in model_choose_list[:2]]))
        model_example2 = '...\n'
        model_example3 = ''.join(['{}\n'.format(model) for model in model_choose_list[-1:]])
        model_example = model_example + model_example2 + model_example3
        suffix += model_example

    style = '''\n Please do not include anything other than the framework list in your response and follow the rules below:
                        1. Do not use capital letters.
                        2. Do not privode operators other than the specified GNN operators and the activation function operators.
                        3. Strictly adhere to the format shown in the example above.
                        4. Do not provide duplicate frameworks'''
    suffix += style
    if cfg.model.type == 'pfgnas':
        if (stage == 0):
            return user_input + notice1 + suffix
        elif (stage < 4):  # Exploration phase
            return user_input + pfgnas_experiments_prompt(models_res_list, cfg.data.type) + notice1 + suffix
        else:  # Exploitation phase
            return user_input + pfgnas_experiments_prompt(models_res_list, cfg.data.type) + notice2 + suffix

    else:
        if (stage == 0):
            return user_input + notice1 + suffix
        elif (stage < 4):  #Exploration phase
            return user_input + experiments_prompt(models_res_list, cfg.data.type) + notice1 + suffix
        else: # Exploitation phase
            return user_input + experiments_prompt(models_res_list, cfg.data.type) + notice2 + suffix

def main_prompt_ablation(cfg,struct_dict,ablation, models_res_list=None,stage=0):
    if cfg.model.type == 'pfgnas':
        Search_Task= f'A federated scenario refers to a situation where there are multiple clients (for example, multiple organizations, devices, or users) working together to achieve a goal, but there may be a need for data isolation and privacy protection between them. ' \
                     f'In a federated graph scenario, with {cfg.federate.client_num} clients, each holding a segment of the {cfg.data.type} graph dataset, the task is to customize the choice of the best sub-model for each client within the search space. This is done with the goal of maximizing accuracy and ROC-AUC values across all clients. '
    else:
        Search_Task =f'The task is to choose the best GNN architecture and the best activation function on a given dataset. The model will be trained and tested on {cfg.data.type}. '

    Search_Space = get_search_space(struct_dict,link=None, comprehensive = False)

    if cfg.model.type == 'pfgnas':
        Search_Strategy = '''Once again, your task is to search the optimal personalized federated models, this is, help all client search the optimal intermediate operations of sub-models on a given experimental dataset. 
        The search strategy we set encompasses two stages: 
        1. Exploration Stage: in the initial stage of the search, you should focus more on exploring the entire search space randomly, rather than just focusing on the current local optimal results. 
        2. Exploitation Stage: with a certain amount of experimental results, in the exploitation stage, you can iteratively refine the search for the best operation lists by querying new lists based on existing ones and their corresponding performance. You should employ optimization algorithms such as Bayesian optimization or evolutionary algorithms to more effectively explore the optimal combinations of operations instead of random selection.'''
    else:
        Search_Strategy = '''Once again, your task is to help me find the optimal combination of operations while specifying experimental dataset. The main difficulty of this task is how to reasonably arrange the selection strategy of the operation list, and each selected operation list corresponds to the highest accuracy that the operation can achieve. You should select a new operation list to query based on the existing operation lists and their corresponding accuracy, in order to iteratively find the best operation list.

            At the beginning, when there were few experimental results, we in the Exploration phase, you need to explore the operation space and identify which operation lists are promising. You can #randomly# select a batch of operation lists corresponding to each layer and evaluate their performance. Afterwards, we can sort the operation lists based on their accuracy and select some well performing operation lists as candidates for our Exploitation phase.
            
            When we have a certain amount of experimental results, we are in the Exploitation phase, you need focus on improving search by exploring the operating space more effectively. You can use optimization algorithms, such as Bayesian optimization or Evolutionary algorithm, to search for the best combination of operations, rather than randomly selecting the list of operations.
            '''
    # Part 1: explain  GNAS
    if ablation == '1':
        user_input = Search_Space + Search_Strategy
    elif ablation == '2':
        user_input = Search_Task + Search_Strategy
    elif ablation == '3':
        user_input = Search_Task + Search_Space 
    else:
        user_input = Search_Task + Search_Space + Search_Strategy

    # Exploration phase
    notice1 = '''\n# Now, it should be in the Exploration stage. You can randomly select from operators, considering some well-performing lists as candidates for the exploitation stage.#\n\n'''
    # Exploitation phase
    notice2 = '''\n#Due to the availability of a certain number of experimental results, it is currently in the Exploitation stage. You should choose nearby samples that temporarily display the best results for searching, especially those that rank in the top 10% or 20% of existing experimental results. At the same time, you should try to avoid exploring sample structures with poor results, which can cause waste.#\n\n'''
    if cfg.model.type == 'pfgnas':
        model_choose_list = generate_federated_models(cfg.federate.client_num, num_models=5)

        # suffix = ''' Please do not include anything other than the model list in your response. For simplicity, please use the following #9 letters# in the alphabet to represent GNN operations:
        # #'a':'gcn', 'b':'sage', 'c':'gpr', 'd':'gat', 'e': 'gin', 'f':'fc', 'g':'sgc', 'h':'arma', 'i':'appnp','j':'identity', 'k':'zero'# ,
        # and represent activation function operations with these #5 numbers#:
        # #'1':'sigmoid','2':'tanh', '3':'relu', '4':'linear', '5':'elu' #.
        # Each sub-model for each client should include #4 operations# : the first three operations are GNN operations, and the fourth is an activation function operation, such as 'aaa2'.
        # And each federated model is composed of sub-models from #{} clients#, for example, a federated model can be represented as a list: {}. In your response, adhere strictly to the format exemplified below, for example:'''.format(cfg.federate.client_num, model_choose_list[0].split(':')[-1])

        suffix = ''' there are 11 GNN operators: ['gcn', 'sage', 'gpr', 'gat',  'gin', 'fc', 'sgc', 'arma', 'appnp', 'identity', 'zero'], and there are 5 activation function operators: ['sigmoid', 'tanh', 'relu', 'linear', 'elu' ]. 
               Each sub-architecture for each client should include #4 operators# : the first three operators are GNN operations, and the fourth is an activation function operator, such as ['fc', 'fc', 'gpr', 'sigmoid']. 
               And each federated framework is composed of #{} sub-architectures#, for example, a federated framework can be represented as a list: {}. In your response, adhere strictly to the format exemplified below, for example:'''.format(
            cfg.federate.client_num, model_choose_list[0].split(':')[-1])

        model_example = ''' Assuming there are {} clients in each federated scenario, please provide #5# different personalized federated framework. For example:\n{}'''.format(cfg.federate.client_num,
             ''.join(['{}\n'.format(model) for model in model_choose_list[:2]]))
        model_example2 = '...\n'
        model_example3 = ''.join(['{}\n'.format(model) for model in model_choose_list[-1:]])
        model_example = model_example + model_example2 + model_example3
        suffix += model_example

        #model_choose_list = '1. client1-client2-client3: bfg3-ccf4-dac1',
        # '2. client1-client2-client3: ifh5-aci1-egi2',
        # '3. ...
        #  ...
        # '10. client1-client2-client3: ahh4-caf3-hfe5'

    else:
        model_choose_list =random_global_model_list(num_models=10)
        # suffix = '''Please do not include anything other than the operation list in your response.
        #     And you should give 10 different models at a time, one model contains #4# operations. The first three are chosen from GNN operations (i.e., 'a':'gcn', 'b':'sage', 'c':'gpr', 'd':'gat','e': 'gin', 'f':'fc', 'g':'sgc', 'h':'arma', 'i':'appnp','j':'identity', 'k':'zero'), and the last one is chosen from activation function operation('1':'sigmoid','2':'tanh', '3':'relu', '4':'linear','5': 'elu'), chosen from the search space described above. Please do not give repeated models.
        #     Please represent the selected operations using those key(i.e.,letters and numbers). Your response only need include the operation list, for example:
        #     1: edb2
        #     2: cci1
        #     3...
        #     ......
        #     10: fha5.
        #     And The response you give must strictly follow the format of this example. '''
        suffix = '''there are 11 GNN operators: ['gcn', 'sage', 'gpr', 'gat',  'gin', 'fc', 'sgc', 'arma', 'appnp', 'identity', 'zero'], and there are 5 activation function operators: ['sigmoid', 'tanh', 'relu', 'linear', 'elu' ]. 
         Each architecture should include #4 operators# : the first three operators are GNN operations, and the fourth is an activation function operator, such as {}. In your response, adhere strictly to the format exemplified below, for example:'''\
            .format(model_choose_list[5].split(':')[-1])
        model_example = ''' Please provide #10# different personalized federated framework. For example:\n{}'''.format(
            ''.join(['{}\n'.format(model) for model in model_choose_list[:2]]))
        model_example2 = '...\n'
        model_example3 = ''.join(['{}\n'.format(model) for model in model_choose_list[-1:]])
        model_example = model_example + model_example2 + model_example3
        suffix += model_example

    style = '''\n Please do not include anything other than the framework list in your response and follow the rules below:
                        1. Do not use capital letters.
                        2. Do not privode operators other than the specified GNN operators and the activation function operators.
                        3. Strictly adhere to the format shown in the example above.
                        4. Do not provide duplicate frameworks'''
    suffix += style
    if cfg.model.type == 'pfgnas':
        if ablation == '4':
            return user_input + notice1 + suffix
        else:
        
            if (stage == 0):
                return user_input + notice1 + suffix
            elif (stage < 4):  # Exploration phase
                return user_input + pfgnas_experiments_prompt(models_res_list, cfg.data.type) + notice1 + suffix
            else:  # Exploitation phase
                return user_input + pfgnas_experiments_prompt(models_res_list, cfg.data.type) + notice2 + suffix

    else:
        if (stage == 0):
            return user_input + notice1 + suffix
        elif (stage < 4):  #Exploration phase
            return user_input + experiments_prompt(models_res_list, cfg.data.type) + notice1 + suffix
        else: # Exploitation phase
            return user_input + experiments_prompt(models_res_list, cfg.data.type) + notice2 + suffix


def experiments_prompt(models_res_list, dataname):
    # models_res_list = [{'abcd':{"acc":0.1233,"roc_auc":0.2333},'cdff':{'acc':0.4354,'roc_auc':0.9843},
    #                     'dfed':{'acc':0.4354,'roc_auc':0.4875},'abdc':{'acc':0.3671,'roc_auc':0.8940},
    #                     'bcda':{'acc':0.4258,'roc_auc':0.1238},'fcab':{'acc':0.1584,'roc_auc':0.2594},
    #                     'aaaa':{'acc':0.9215,'roc_auc':0.6127},'bfda':{'acc':0.1565,'roc_auc':0.0534},
    #                     'aeef':{'acc':0.4354,'roc_auc':0.9587},'edbb':{'acc':0.1259,'roc_auc':0.3207}},
    #                    {'bcda': {"acc": 0.2786, "roc_auc": 2855}, 'cdfa': {'acc': 0.1525, 'roc_auc': 0.2687},
    #                     'efba': {'acc': 0.4354, 'roc_auc': 0.4875}, 'ffdc': {'acc': 0.2561, 'roc_auc': 0.3856},
    #                     'edcb': {'acc': 0.1575, 'roc_auc': 0.5377}, 'bfab': {'acc': 0.1584, 'roc_auc': 0.2568},
    #                     'aaaa': {'acc': 0.3584, 'roc_auc': 0.4686}, 'bbbb': {'acc': 0.3968, 'roc_auc': 0.9687},
    #                     'eeee': {'acc': 0.3434, 'roc_auc': 0.4468}, 'cccc': {'acc': 0.2534, 'roc_auc': 0.6867}},
    #                    ]
    last_round_res = models_res_list[-1]
    metrics = ['acc','roc_auc']
    avg_dict = extarct_metric(last_round_res,metrics)
    all_dicts, duplicates_set = extarct_all(models_res_list, metrics)
    if (len(models_res_list) < 2):
        sorted_all_dict = dict(sorted(all_dicts.items(), key=lambda x: (x[1]['acc'], x[1]['roc_auc']), reverse=True))
        all_dict_str = _dict_to_string(sorted_all_dict)# iteration<2
        # prompt_history = '''#You can refer to the all historical experiment results. These experiment results were generated using the operation combinations you provided as follows, including accuracy(i.e., acc) and AUC-ROC(i.e., roc_auc) values:\n{}  I hope you can learn the commonalities between the well performing models to achieve better results #and avoid the mistakes of poor models to avoid achieving such poor results again.#\n''' \
        #     .format(''.join(all_dict_str))
        prompt_history = '''#For simplicity, we use those keys represent GNN operators and activation function operators: 'a':'gcn', 'b':'sage', 'c':'gpr', 'd':'gat', 'e': 'gin', 'f':'fc', 'g':'sgc', 'h':'arma', 'i':'appnp','j':'identity', 'k':'zero','1':'sigmoid','2':'tanh', '3':'relu', '4':'linear','5': 'elu'. You can refer to the historical experiment results.
           The currently proposed frameworks include:\n{}\n The current top five models with the best performance are: \n{} I hope you can learn the commonalities between the well performing models to achieve better results #and avoid the mistakes of poor models to avoid achieving such poor results again.#\n''' \
            .format(list(all_dicts.keys()), ''.join(all_dict_str))
    else:
        last_dicts, _ = extarct_all([last_round_res], metrics)
        sorted_last_dict = dict(sorted(last_dicts.items(), key=lambda x: (x[1]['acc'], x[1]['roc_auc']), reverse=True))
        # 提取top3
        top3_items = list(sorted_last_dict.items())[:3]
        top3_dict = dict(top3_items)
        last_dict_str = _dict_to_string(top3_dict)
        # prompt_history = '''#You can refer to the all historical experiment results. These experiment results were generated using the operation combinations you provided as follows, including accuracy(i.e., acc) and AUC-ROC(i.e., roc_auc) values:\n{}  I hope you can learn the commonalities between the well performing models to achieve better results #and avoid the mistakes of poor models to avoid achieving such poor results again.#\n''' \
        #     .format(''.join(last_dict_str))
        prompt_history = '''#For simplicity, we use those keys represent GNN operators and activation function operators: 'a':'gcn', 'b':'sage', 'c':'gpr', 'd':'gat', 'e': 'gin', 'f':'fc', 'g':'sgc', 'h':'arma', 'i':'appnp','j':'identity', 'k':'zero','1':'sigmoid','2':'tanh', '3':'relu', '4':'linear','5': 'elu'. You can refer to the historical experiment results, and 
            The currently proposed frameworks include:\n{}\n The current top 3 models with the best performance are: \n{} I hope you can learn the commonalities between the well performing models to achieve better results #and avoid the mistakes of poor models to avoid achieving such poor results again.#\n''' \
            .format(list(all_dicts.keys()), ''.join(last_dict_str))

    # prompt_repeat = '''In the above experimental results, there are some repetitive combinations of operations, as follows:{}.'''.format(''.join(
    #     [f'{get_model_name(arch)},' for arch in duplicates_list]))

    prompt_repeat = '''In the above experimental results, there are some repetitive combinations of operations, as follows:{}.'''.format(
        ''.join(
            [f'{arch},' for arch in duplicates_set]))

    prompt_repeat += '''\nThe combinations you propose should be strictly #different# from the structure of the existing experimental results.#You should not raise the combinations that are already present in the above experimental results again.#\n'''
    prompt_avg_res = f'''\nPlease propose 10 better and #different# combinations of operations with accuracy strictly greater than {avg_dict['acc']}, while also having an auc_roc value strictly greater than {avg_dict['roc_auc']} on {dataname} dataset.\n'''
    return prompt_history + prompt_repeat + prompt_avg_res


def pfgnas_experiments_prompt(models_res_list, dataname):

    last_round_res = models_res_list[-1]
    avg_dict,_ = calculate_average_dict(last_round_res)
    all_dicts,duplicates_set = pf_extarct_all(models_res_list)
    last_dicts,_ = pf_extarct_all([last_round_res])
    sorted_all_dict = sort_dicts_by_test_metrics(all_dicts,reverse=True)
    first_five_keys = list( sorted_all_dict.keys())[:5]
    # print("前五位键：", first_five_keys)
    sorted_last_dict = sort_dicts_by_test_metrics(last_dicts,reverse=True)#
    result_list = dict_to_string(sorted_all_dict)
    # last_dict_str = dict_to_string(sorted_last_dict)
    if(len(models_res_list) == 1): # iteration<2
        prompt_history = '''#For simplicity, we use those keys represent GNN operators and activation function operators: 'a':'gcn', 'b':'sage', 'c':'gpr', 'd':'gat', 'e': 'gin', 'f':'fc', 'g':'sgc', 'h':'arma', 'i':'appnp','j':'identity', 'k':'zero','1':'sigmoid','2':'tanh', '3':'relu', '4':'linear','5': 'elu'. You can refer to the historical experiment results, and 
    The currently proposed frameworks include:\n{}\n The current top five models with the best performance are: \n{} I hope you can learn the commonalities between the well performing models to achieve better results #and avoid the mistakes of poor models to avoid achieving such poor results again.#\n''' \
            .format(list(all_dicts.keys()), ''.join([f'{res}' for res in result_list]))
    else:
        prompt_history = '''#For simplicity, we use keys represent GNN operators and activation function operators: 'a':'gcn', 'b':'sage', 'c':'gpr', 'd':'gat', 'e': 'gin', 'f':'fc', 'g':'sgc', 'h':'arma', 'i':'appnp','j':'identity', 'k':'zero','1':'sigmoid','2':'tanh', '3':'relu', '4':'linear','5': 'elu'. You can refer to the historical experiment results, and 
            The currently tested frameworks include:\n{}\n The current top five models with the best performance are: \n{} I hope you can learn the commonalities between the well performing models to achieve better results #and avoid the mistakes of poor models to avoid achieving such poor results again.#\n''' \
            .format(list(all_dicts.keys()), ''.join([f'{res}' for res in result_list[:5]]))

    # else:
    #     prompt_history = '''#For simplicity, we use keys represent GNN operators and activation function operators: 'a':'gcn', 'b':'sage', 'c':'gpr', 'd':'gat', 'e': 'gin', 'f':'fc', 'g':'sgc', 'h':'arma', 'i':'appnp','j':'identity', 'k':'zero','1':'sigmoid','2':'tanh', '3':'relu', '4':'linear','5': 'elu'. You can refer to the historical experiment results. These experiment results were generated using the operation combinations you provided as follows, including accuracy(i.e., acc) and AUC-ROC(i.e., roc_auc) values:\n{}  I hope you can learn the commonalities between the well performing models to achieve better results #and avoid the mistakes of poor models to avoid achieving such poor results again.#\n''' \
    #         .format(''.join(last_dict_str))
    # prompt_repeat = '''In the above experimental results, there are some repetitive combinations of operations, as follows:{}.'''.format(''.join(
    #     [f'{arch},' for arch in duplicates_set]))
    prompt_repeat = '''\nThe frameworks you propose should be strictly #different# from the structure of the existing experimental results. #You should not raise the combinations that are already present in the above experimental results again.#\n'''
    prompt_avg_res = f'''\nPlease provide 10 better and #different# frameworks with accuracy strictly greater than {avg_dict['0']['test_acc']},  while also having an ROC_AUC value strictly greater than {avg_dict['0']['test_roc_auc']} on {dataname} dataset.\n'''

    return prompt_history + prompt_repeat + prompt_avg_res


if __name__ == '__main__':
    # models_res_list = [{
    # 'aecd-bcda-cdfa': {'2':{"train_acc":0.4233,"train_roc_auc":0.2566},'1': {"train_acc": 0.2786, "train_roc_auc": 0.8115}, '3': {'train_acc': 0.1525, 'train_roc_auc': 0.2687},'0':{'test_acc':0.5115,'test_roc_auc':0.8653}},
    # 'abcd-bdda-cdfa': {'1': {"train_acc": 0.1233, "train_roc_auc": 0.2333},'2': {"train_acc": 0.2786, "train_roc_auc": 0.855},'3': {'train_acc': 0.1685, 'train_roc_auc': 0.1327},'0': {'test_acc': 0.1555, 'test_roc_auc': 0.5723}},
    # 'abcd-bcda-crta': {'1': {"train_acc": 0.1233, "train_roc_auc": 0.2643},'2': {"train_acc": 0.2786, "train_roc_auc": 0.5587},'3': {'train_acc': 0.1525, 'train_roc_auc': 0.4667},'0': {'test_acc': 0.5255, 'test_roc_auc': 0.4323}},
    # 'abcd-bcda-cgda': {'1': {"train_acc": 0.1233, "train_roc_auc": 0.6542},'2': {"train_acc": 0.4755, "train_roc_auc": 0.855}, '3': {'train_acc': 0.3642, 'train_roc_auc': 0.9637},'0': {'test_acc': 0.6555, 'test_roc_auc': 0.3923}},
    # 'cgda-bcda-ctra': {'1': {"train_acc": 0.1233, "train_roc_auc": 0.6843},'2': {"train_acc": 0.5755, "train_roc_auc": 0.855},'3': {'train_acc': 0.1525, 'train_roc_auc': 0.6924}, '0': {'test_acc': 0.8155, 'test_roc_auc': 0.2623}}
    # },
    # {
    # 'abcd-bcda-cdfa': {'1': {"train_acc": 0.7561, "train_roc_auc": 0.2968},'2': {"train_acc": 0.2786, "train_roc_auc": 0.8676},'3': {'train_acc': 0.6278, 'train_roc_auc': 0.2676},'0': {'test_acc': 0.6155, 'test_roc_auc': 0.8623}},
    # 'abad-bcdd-cdca': {'1': {"train_acc": 0.1233, "train_roc_auc": 0.2333},'2': {"train_acc": 0.2786, "train_roc_auc": 0.7675},'3': {'train_acc': 0.1525, 'train_roc_auc': 0.4947},'0': {'test_acc': 0.5345, 'test_roc_auc': 0.2723}},
    # 'aaad-yedy-ctew': {'1': {"train_acc": 0.5777, "train_roc_auc": 0.2645},'2': {"train_acc": 0.4767, "train_roc_auc": 0.4645},'3': {'train_acc': 0.1258, 'train_roc_auc': 0.2546},'0': {'test_acc': 0.5575, 'test_roc_auc': 0.2823}},
    # 'acfd-dyye-gdgf': {'1': {"train_acc": 0.1223, "train_roc_auc": 0.4577},'2': {"train_acc": 0.2786, "train_roc_auc": 0.4746}, '3': {'train_acc': 0.1544, 'train_roc_auc': 0.2676}, '0': {'test_acc': 0.5255, 'test_roc_auc': 0.4623}},
    # 'ared-strt-atga': {'2': {"train_acc": 0.7458, "train_roc_auc": 0.2145},'1': {"train_acc": 0.2786, "train_roc_auc": 0.1634}, '3': {'train_acc': 0.5946, 'train_roc_auc': 0.24487}, '0': {'test_acc': 0.4555, 'test_roc_auc': 0.7823}}
    #  }]

    file_path = '../../../exp/FedAvg_pfgnas_cora_10/models_res_list.log'
    with open(file_path, 'r') as f:
        data = f.readlines()
    models_res_list = [json.loads(line.strip()) for line in data]
    dataname = 'cora'
    text = pfgnas_experiments_prompt(models_res_list, dataname)
    print(text)
    # model_choose_list = generate_federated_models(5, num_models=20,operation_num=4)
    # print( model_choose_list)