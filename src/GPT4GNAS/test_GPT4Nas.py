import requests
import json
#
# url = 'https://896f7c04.gpt4-58r.pages.dev/'
# url = 'https://gpt4-58r.pages.dev/v1/chat/completions'
# url = 'https://openai.eaglecode-fh.uk/v1/chat/completions'
# url = 'https://openapi.896f7c04.gpt4-58r.pages.dev/v1/chat/completions '
# url = 'https://chatgptproxyapi-d8n.pages.dev/api/v1/chat/completions'

# my
# url = "https://openai.1rmb.tk/v1/chat/completions"
# api_key = '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'

# zx
api_key = 'sb-566f675f2973f55bcd91071eebe87aa20071fd67f158b9f8'  # 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'
url = "https://api.openai-sb.com/v1/chat/completions"

headers = {
  'Authorization': f'Bearer {api_key}',
  'Content-Type': 'application/json'
}
# content = '''I started to write an academic paper for IJCAI conference, the title is 'Personalized Federated Learning with Graph Neural Architecture Search Empowered by LLM Enhancement', now I have finished the introduction part,  you should put forward detailed and specific revision suggestions, and finally provide the modified version, and the introduction as follows:\n
#           With the emphasis on data security and user privacy, federated graph machine learning (FGML) as a distributed learning paradigm that trains graph machine learning models on decentralized data has attracted much attention in a wide diversity of real-world domain recently,e.g., health-care, transportation, bioinformatics, and recommendation systems. \n
# However, one important problem FL concerns is data distribution heterogeneity, since the decentralized data, collected by different institutes using different methods and aiming at different tasks, are highly likely to follow non-identical distributions(non-IID). The traditional FGML ignore the inevitable data heterogeneity between clients with FedAvg~\cite{}, resulting in the slowly convergence and degrading accuracy. To learn client-specific personalized models in the FL setting, recent advancements in personalized federated learning (PFL) mainly discuss two major solutions, namely group-wise~\cite{} and individual-wise~\cite{}. Group-wise PFL assumes the clients with similar data distribution can be clustered in a group and they share the same model parameters. However, the clustering result is significantly influenced by the latest gradients from clients, which are usually unstable during local training. Therefore, individual-wise PFL provides a more generalized assumption, namely each client’s data distribution is different from others, therefore, those methods concentrate on designing individual model for each client. Despite their success, these techniques greatly reduce the labour of human experts and can find better GNN architectures than the human-invented ones.\n
# To automatically design more directed graph neural networks (GNNs) for different graph data distribution, graph neural architecture search (NAS) has been utilized to search for an optimal GNN architecture~\cite{}. Existing  GNN NAS methods can be categorized into three types according to their search strategies, i.e., reinforcement learning GNAS~\cite{}, differential gradient GNAS~\cite{}, and evolutionary GNAS~\cite{}.\n
# However, the design of graph neural architecture search algorithms requires heavy manual work with domain knowledge. To solve this issue, GENIUS first investigate the potential of the large language model (LLM) to perform Neural Architecture Search on the design of Convolution Neural Networks (CNNs). Moreover, GPT4GNAS utilize GPT-4 to guide GPT-4 toward the generative task of graph neural architectures.  However, they are not applicable to the federated learning scenarios with distributed and private graph datasets, which reduce their practicality. In most tasks when the network architecture is not determined a priori, it remains very difficult to search for the optimal network architectures and train them efficiently in a decentralized setup. This requires GNN NAS techniques to be able to cope with the federated learning scenarios, i.e., learn the optimal GNN architecture in a distributed and privacy-preserving manner. Even thought some researchers notice this problem, e.g., FedNAS, DFNAS and FL-AGNNS, those work aim to find a global GNN model instead of recommending different GNN models for different clients.\n
# To bridge this gap, we design a \textbf{P}ersonalized \textbf{F}ederated learning with \textbf{G}raph \textbf{N}eural \textbf{A}rchitecture \textbf{S}earch (PFGNAS). Inspired by GENIUS, we enables a LLM as an agents to cooperatively design the high-performance GNN model while keeping personal information on local devices.  But this direct solution would introduce two technical challenges: first, how to guide the LLM model to search for architecture within the federated framework. Because LLM as an nlp model, there is a gap between it and graph architecture search. And second, how to optimize personalized models on each client, since each model owns different architecture and parameters, directly applying these techniques to the current clients sides in GNAS will result in consistency collapse issues in server sides.\n
#
# To tackle all the above-mentioned challenges, in PFGNAS, we design a federated GNAS optimization prompt, it can harness the historic knowledge and powerful learning capability of LLMs as an experienced controller  to consider the GNN architectures favored by each client.\n
# Furthermore,  instead of training one by one thousands of separate models from scratch, we build a single large network (supernet) capable of emulating any architecture in the search space,  so that the architecture can be optimized with respect to its validation set performance by LLMs.  The supernet adopt weight sharing strategy to reduce the computational cost and solve the issue of alignment problem in the clients. We validate the proposed methodology on three datasets, and our experimental results affirm its capacity to provide superior performance compared to existing methods.\n
# Our main contributions are summarized as follows:\n
# We first apply GNAS to personalized federated learning, conducting architecture search adaptively based on the features of each client.
# We address the graph architecture search problem in federated learning for the LLM model and explore the capabilities of LLM.
# We devise a novel optimization scheme for personalized models, enhancing search efficiency and resolving the issue of client information fusion consistency.
# We conduct extensive experiments, and the results validate the effectiveness of our approach.
# The rest of the paper is organized as follows. In Section~\ref{section:Related Work}, we review related work. In Section~\ref{section:Problem Formulation}, we introduce some preliminary concepts and formalize the problem. Then in Section~\ref{section:Methodology}, we ??????. Experiments and further analysis are presented in Section~\ref{section:Experiments}. Finally, we conclude the work in Section~\ref{section:Conclusion}.'''

content = '给我讲一个笑话'
payload = {
  "model": "gpt-4-1106-preview",
  "messages": [
    {
      "role": "user",
      "content": content
    }
  ]
}

response = requests.post(url, headers=headers, json=payload)
# response.raise_for_status() # 抛出异常，如果响应码不是200
data = response.json()
# print(data)
output_text = data['choices'][0]['message']['content']
print(output_text)

# try:
#     response = requests.post(url, headers=headers, json=payload)
#     response.raise_for_status() # 抛出异常，如果响应码不是200
#     data = response.json()
#     # print(data)
#     output_text = data['choices'][0]['message']['content']
#     print(output_text)
# except requests.exceptions.RequestException as e:
#     print(f"请求错误: {e}")
# except json.JSONDecodeError as e:
#     print(f"无效的 JSON 响应: {e}")
