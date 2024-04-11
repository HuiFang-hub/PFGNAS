import numpy as np
import torch
from torch_geometric.datasets import Planetoid, Reddit
from torch_geometric.data import ClusterData, Data, ClusterLoader
from numpy.random import randint, random
import pickle
import logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

OBJ_END = 'eofeofeof'


def change_machine_of_data(data, id=''):
    return


IP_PORT_BASE = 6000
ADDR = "addr"


def reset_ipport():
    for label in ['', '0', '1', '2', '3', '4', '5', '6', '7']:
        path = '{}/ip_oprt{}.pkl'.format(ADDR, label)
        with open(path, 'wb') as f:
            pickle.dump(('localhost', IP_PORT_BASE + 100), f)
    return


def read_ipport():
    for label in ['', '0', '1', '2', '3', '4', '5', '6', '7']:
        path = '{}/ip_oprt{}.pkl'.format(ADDR, label)
        with open(path, 'rb') as f:
            print(pickle.load(f))
    return


def get_ip_port(id=''):
    path = '{}/ip_oprt{}.pkl'.format(ADDR, id)
    with open(path, 'rb') as f:
        ans = pickle.load(f)
    if ans[1] >= IP_PORT_BASE + 1000:
        new_ip_port = (ans[0], IP_PORT_BASE)
    else:
        new_ip_port = (ans[0], ans[1] + 100)
    with open(path, 'wb') as f:
        pickle.dump(new_ip_port, f)
    return ans


def global_evaluate(model, dataset_name, mode='test'):
    '''
    test the accuracy on the whole dataset
    :param model: nn.Module
    :param dataset_name: str
    :param mode: val or test
    :return:
    '''
    model.eval()
    data = get_raw_data(dataset_name)[0]
    mask = eval('data.{}_mask'.format(mode))
    print(num_correct(model(data.x, data.edge_index)[mask], data.y[mask]))
    return accuracy(model(data.x, data.edge_index)[mask], data.y[mask])


def socket_recv_with_response(socket):
    '''
    recv message with specified socket and response okk
    :param socket:
    :return: decoded data
    '''
    data = b''
    while True:
        packet = socket.recv(1024*4)
        # if packet.__contains__(OBJ_END.encode()):
        #     data += packet.replace(OBJ_END.encode(), b'')
        #     break
        data += packet
        if data[-9:].__contains__(OBJ_END.encode()):
            data = data[:-9]
            break
    socket.send('okk'.encode())
    if data.__sizeof__() >= 5*1024*1024:
        # logger.info('recv big obj:{:.2f}M'.format(
        #     data.__sizeof__()/(1024*1024)))
        print('recv big obj:{:.2f}M'.format(data.__sizeof__()/(1024*1024)))
    return pickle.loads(data)


def socket_send_with_waiting_response(socket, mess):
    '''
    send out message and waiting for okk as response
    :param socket:
    :param mess:
    :return:
    '''
    mess = pickle.dumps(mess)
    socket.send(mess+OBJ_END.encode())
    socket.recv(1024)
    return


def socket_recv(socket):
    '''
    only recv message
    not send okk
    :param socket:
    :return:
    '''
    data = b''
    while True:
        packet = socket.recv(1024*4)
        # if packet.__contains__(OBJ_END.encode()):
        #     data += packet.replace(OBJ_END.encode(), b'')
        #     break
        data += packet
        if data[-9:].__contains__(OBJ_END.encode()):
            data = data[:-9]
            break
    if data.__sizeof__() >= 5*1024*1024:
        # logger.info('recv big obj:{:.2f}M'.format(
        #     data.__sizeof__()/(1024*1024)))
        print('recv big obj:{:.2f}M'.format(data.__sizeof__()/(1024*1024)))
    return pickle.loads(data)


def socket_send(socket, mess):
    '''
    only send message
    not waiting for response
    :param socket:
    :param mess:
    :return:
    '''
    mess = pickle.dumps(mess)
    socket.send(mess+OBJ_END.encode())
    return


def serialize_model(model):
    '''
    encode all the parameters in a nn model
    :param model: nn.Module
    :return: dic
    '''
    param_dict = {}
    for name, param in model.named_parameters():
        param_dict[name] = param
    # print(param_dict)
    return param_dict


def analyse_subgraph(datas):
    """
    analyse the statistics of graphs
    :param datas: a list of torch_geometric.data.Data
    :return:
    """
    for i in range(len(datas)):
        print('num_node', len(eval('datas[{}]'.format(i)).train_mask))
        print('num_edge', len(eval('datas[{}]'.format(i)).edge_index[0]))
        print('num of train node', torch.sum(
            eval('datas[{}]'.format(i)).train_mask))
        print('num of val node', torch.sum(
            eval('datas[{}]'.format(i)).val_mask))
        print('num of test node', torch.sum(
            eval('datas[{}]'.format(i)).test_mask))
        print()
    return


def load_data(dataset='Cora'):
    '''
    get the num_features,num_classes,edge_index,x,y,train_mask,test_mask of a dataset
    :param dataset:
    :return:
    '''
    if dataset.lower() in ['cora', 'citeseer', 'pubmed']:
        dataset = Planetoid(root='data/', name=dataset)
    elif dataset.lower() == 'reddit':
        dataset = Reddit('data/reddit/')
    if isinstance(dataset, str):
        return None
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    data = dataset[0].to(device)
    return dataset.num_node_features, dataset.num_classes, data.edge_index, data.x, data.y, data.train_mask, data.test_mask


def get_raw_data(dataset='cora'):
    '''
    get raw dataset
    :param dataset:
    :return: Planetoid or Reddit and dataset[0] is type of torch_geometric.data.Data
    '''
    if dataset.lower() in ['cora', 'citeseer', 'pubmed']:
        dataset = Planetoid(root='data/', name=dataset)
    elif dataset.lower() == 'reddit':
        dataset = Reddit('data/reddit/')
    return dataset


def cal_edge_attr_for_gmmconv(edge_index):
    '''
    contruct edge_attr on specified dataset for gmmconv
    :param edge_index:
    :return:
    '''
    message = edge_index[0]
    ans = torch.ones(torch.max(edge_index) + 1)
    for index in message:
        ans[index] += 1
    edge_attr = []
    for i in range(len(edge_index[0])):
        u, v = edge_index[0][i], edge_index[1][i]
        edge_attr.append([1/np.sqrt(ans[u]), 1/np.sqrt(ans[v])])
    # device = torch.device('cpu')
    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    return torch.Tensor(edge_attr).to(edge_index.device)


class PartitionTool():
    def __init__(self):
        return

    def partition_subgraph(self, data, k, copy_node=False):
        '''
        partition a data into k parts
        :param data: torch_geometric.data.Data
        :param k:
        :param copy_node:
        :return:
        '''
        cluster = ClusterData(data, k)
        print(len(cluster.perm))
        # cluster.perm = torch.Tensor(np.random.shuffle(np.array(cluster.perm)))
        # return
        # cluster.perm =
        for i in range(k):
            exec('self.idx_{} = cluster.perm[cluster.partptr[{}]:cluster.partptr[{}]]'.format(
                i, i, i + 1))
        print('cluster over!')
        # expand
        if copy_node == True:
            def expand(edge_index, idx):
                new_edge_index_u = []
                new_edge_index_v = []
                extra_node = set()
                for tmp in idx:
                    for v in v2e[int(tmp)]:
                        extra_node.add(v)
                        extra_node.add(int(tmp))
                    new_edge_index_u += [int(tmp)
                                         for i in range(len(v2e[int(tmp)]))]
                    new_edge_index_v += v2e[int(tmp)]
                # print(len(id_set))
                print('over for loop')
                return torch.LongTensor(list(extra_node)), torch.LongTensor([new_edge_index_u, new_edge_index_v])
        else:
            def expand(edge_index, idx):
                new_edge_index_u = []
                new_edge_index_v = []
                id_set = set()
                for tmp in idx:
                    id_set.add(int(tmp))
                for tmp in idx:
                    for v in v2e[int(tmp)]:
                        if id_set.__contains__(v):
                            new_edge_index_u.append(int(tmp))
                            new_edge_index_v.append(v)
                return torch.LongTensor(list(idx)), torch.LongTensor([new_edge_index_u, new_edge_index_v])

        edge_index = np.array(data.edge_index)
        v2e = {}
        # for i in range(len(edge_index[0])):
        #     u, v = edge_index[0][i], edge_index[1][i]
        #     v2e[u] = []
        #     v2e[v] = []
        for u in range(len(data.x)):
            v2e[u] = []
        for i in range(len(edge_index[0])):
            u, v = edge_index[0][i], edge_index[1][i]
            v2e[u].append(v)

        for i in range(k):
            print(i)
            tmp_idx_set = set()
            for tmp in eval('self.idx_{}'.format(i)):
                tmp_idx_set.add(int(tmp))
            exec('self.idx_{}, self.edge_index_{} = expand(edge_index, self.idx_{})'.format(
                i, i, i))
            new_idx_set = set()
            for tmp in eval('self.idx_{}'.format(i)):
                new_idx_set.add(int(tmp))
            exec('self.extra_idx_{} = list(new_idx_set.difference(tmp_idx_set))'.format(i))
        print('subgraph expands over')

        def hashing(idx, edge_index):
            idx = np.array(idx)
            edge_index = np.array(edge_index)
            idx = np.sort(idx)
            idx_map = np.zeros(int(np.max(idx)) + 1)
            for i in range(len(idx)):
                idx_map[int(idx[i])] = i
            # print(idx_map)
            for i in range(len(edge_index[0])):
                edge_index[0][i], edge_index[1][i] = idx_map[int(
                    edge_index[0][i])], idx_map[int(edge_index[1][i])]
            return torch.LongTensor(edge_index)

        def hash_extra_idx(idx, extra_idx):
            idx = np.array(idx)
            idx = np.sort(idx)
            idx_map = np.zeros(int(np.max(idx)) + 1)
            for i in range(len(idx)):
                idx_map[int(idx[i])] = i
            for i in range(len(extra_idx)):
                extra_idx[i] = idx_map[extra_idx[i]]
            return torch.LongTensor(extra_idx)

        for i in range(k):
            print(i)
            exec(
                'self.edge_index_{} = hashing(self.idx_{},self.edge_index_{})'.format(i, i, i))
            exec('self.extra_idx_{} = hash_extra_idx(self.idx_{},self.extra_idx_{})'.format(
                i, i, i))
        print('hash over')
        x = data.x
        y = data.y
        train_mask = data.train_mask
        val_mask = data.val_mask
        test_mask = data.test_mask
        datas = []
        for i in range(k):
            print(i)
            exec('idx_{}_bool_tensor = torch.zeros(len(x)).bool()'.format(i))
            exec(
                'for idx in self.idx_{}:idx_{}_bool_tensor[idx] = True'.format(i, i))
            exec('x_{} = x[idx_{}_bool_tensor]'.format(i, i))
            exec('y_{} = y[idx_{}_bool_tensor]'.format(i, i))
            exec('train_mask_{} = train_mask[idx_{}_bool_tensor]'.format(i, i))
            exec('val_mask_{} = val_mask[idx_{}_bool_tensor]'.format(i, i))
            exec('test_mask_{} = test_mask[idx_{}_bool_tensor]'.format(i, i))

            for tmp in eval('self.extra_idx_{}'.format(i)):
                exec('train_mask_{}[tmp] = False'.format(i))
                exec('val_mask_{}[tmp] = False'.format(i))
                exec('test_mask_{}[tmp] = False'.format(i))

            exec('data_{} = Data(x=x_{},y=y_{},edge_index=self.edge_index_{},train_mask=train_mask_{},val_mask=val_mask_{},test_mask=test_mask_{})'.format(
                i, i, i, i, i, i, i))
            datas.append(eval('data_{}'.format(i)))
        print('sub graph pack over')
        return datas

    # static partition and then save them into file system
    def generate_static_train_data(self):
        '''
        init data file structure is as:
        data/
        data/{cora,citeseer,pubmed,reddit}/raw
        :return:
        all the pkl files will be put in the same dir 'raw'
        '''
        # partition_tool = PartitionTool()
        for name in ['cora', 'citeseer', 'pubmed']:
            dataset = get_raw_data(name)
            copy = True
            print(name, copy)
            datas = self.partition_subgraph(dataset[0], 3, copy_node=copy)
            for id in range(3):
                print(len(datas[id].x))
                print(torch.max(datas[id].edge_index))
                path = 'data/{}/{}_{}copynode.pkl'.format(
                    name, id, '' if copy else 'un')
                with open(path, 'wb') as f:
                    pickle.dump(datas[id], f)
            analyse_subgraph(datas)
            copy = False
            print(name, copy)
            datas = self.partition_subgraph(dataset[0], 3, copy_node=copy)
            for id in range(3):
                path = 'data/{}/{}_{}copynode.pkl'.format(
                    name, id, '' if copy else 'un')
                with open(path, 'wb') as f:
                    pickle.dump(datas[id], f)
            analyse_subgraph(datas)

        dataset = Reddit('data/reddit/')
        datas = self.partition_subgraph(dataset[0], 3, False)
        with open('data/reddit/subg_uncopynode.pkl', 'wb') as f:
            pickle.dump(datas, f)
        analyse_subgraph(datas)

        for i in range(3):
            sub_datas = self.partition_subgraph(datas[i], 100, False)
            with open('data/reddit/subsubg{}_uncopynode.pkl'.format(i), 'wb') as f:
                pickle.dump(sub_datas, f)

        datas = self.partition_subgraph(dataset[0], 3, True)
        with open('data/reddit/subg_copynode.pkl', 'wb') as f:
            pickle.dump(datas, f)
        analyse_subgraph(datas)

        for i in range(3):
            sub_datas = self.partition_subgraph(datas[i], 100, False)
            with open('data/reddit/subsubg{}_copynode.pkl'.format(i), 'wb') as f:
                pickle.dump(sub_datas, f)
        return

    def generate_static_uncopy_data_with_torch_gnn(self):
        '''
        generate train data with lib function of torch_geometric
        :return:
        '''
        for name in ['cora', 'citeseer', 'pubmed']:
            print(name)
            data = get_raw_data(name)[0]
            cluster_data = ClusterData(data, num_parts=3, recursive=False)
            train_loader = ClusterLoader(
                cluster_data, batch_size=1, shuffle=True, num_workers=12)
            id = 0
            for batch in train_loader:
                # print(len(datas[id].x))
                # print(torch.max(datas[id].edge_index))
                path = 'data/{}/{}_uncopynode.pkl'.format(name, id)
                id += 1
                with open(path, 'wb') as f:
                    pickle.dump(batch, f)
                analyse_subgraph([batch])
        print('reddit')
        data = Reddit('data/reddit/')[0]
        cluster_data = ClusterData(data, num_parts=3, recursive=False)
        train_loader = ClusterLoader(
            cluster_data, batch_size=1, shuffle=True, num_workers=12)
        datas = []
        for batch in train_loader:
            datas.append(batch)
        with open('data/reddit/subg_uncopynode.pkl', 'wb') as f:
            pickle.dump(datas, f)
        analyse_subgraph(datas)
        # analyse_subgraph(data)
        for i in range(3):
            print(i)
            cluster_data = ClusterData(
                datas[i], num_parts=100, recursive=False)
            train_loader = ClusterLoader(
                cluster_data, batch_size=1, shuffle=True, num_workers=12)
            sub_datas = []
            for batch in train_loader:
                sub_datas.append(batch)
            with open('data/reddit/subsubg{}_uncopynode.pkl'.format(i), 'wb') as f:
                pickle.dump(sub_datas, f)
            analyse_subgraph(sub_datas)
        return


class Contacter():
    '''
    use this class as a interface to process data recv and send
    '''

    def __init__(self):
        self.socket = None
        return

    def recv_with_res(self):
        data = socket_recv_with_response(self.socket)
        return data

    def recv(self):
        data = socket_recv(self.socket)
        return data

    def send_with_waiting_res(self, mess):
        socket_send_with_waiting_response(self.socket, mess)
        return

    def send(self, mess):
        socket_send(self.socket, mess)
        return


def setalize(supermasks):
    '''
    remove the same supermask, ank keep only one copy for each supermask
    :param supermasks:
    :return:
    '''
    sm_set = set()
    for tmp in supermasks:
        sm_set.add(str(tmp))
    setlized_sms = []
    for tmp in sm_set:
        setlized_sms.append(eval(tmp))
    return setlized_sms


# a group of funcitons used in e-a
maskRange = [[1, 5], [1, 12], [1, 24], [
    1, 36], [1, 48], [1, 60], [1, 72], [1, 5]]


def rand_element(idx):
    zero_p = [0, 0, 1/6, 1/5, 1/4, 1/3, 1/2, 0]
    rand_seed = zero_p[idx]
    if random() < rand_seed:
        return 0
    return randint(maskRange[idx][0], maskRange[idx][1] + 1)


def random_supermask():
    '''
    generate random supermask
    :return:
    '''
    mask = []
    idx = 0
    for r in maskRange:
        mask.append(rand_element(idx))
        idx += 1
    return mask


def cross_over(original, target):
    son = []
    for i in range(len(maskRange)):
        if np.random.random() > 0.5:
            son.append(target[i])
        else:
            son.append(original[i])
    return son


def mutate(original):
    c = 0.1
    son = []
    for i, range in enumerate(maskRange):
        if np.random.random() < c:
            son.append(rand_element(i))
        else:
            son.append(original[i])
    return son


def accuracy(output, labels):
    '''
    get accuracy
    :param output:
    :param labels:
    :return:
    '''
    preds = output.max(1)[1].type_as(labels)
    correct = preds.eq(labels).double()
    correct = correct.sum()
    return correct / len(labels)


def num_correct(output, labels):
    '''
    get the number of correct nodes that has been predicted correctly
    :param output:
    :param labels:
    :return:
    '''
    preds = output.max(1)[1].type_as(labels)
    correct = preds.eq(labels).double()
    correct = correct.sum()
    return int(correct)


def load_pkl(name, id, copy_node=True):
    '''
    get the data on the specified machine
    use as test function
    :param name:
    :param id:
    :param copy_node:
    :return:
    '''
    path = 'data/{}/{}_{}copynode.pkl'.format(name,
                                              id, '' if copy_node else 'un')
    with open(path, 'rb') as f:
        data = pickle.load(f)
    return data


def deserialize_model(serialized_params):
    '''
    use as test
    :param serialized_params:
    :return:
    '''
    return pickle.loads(serialized_params)


def partition_dataset(dataset, k, copy_node=False, data=None):
    '''
    old version for partition dataset
    :param dataset:
    :param k:
    :param copy_node:
    :param data:
    :return:
    '''
    if isinstance(dataset, Reddit):
        path = 'data/reddit/subg_copynode.pkl' if copy_node else 'data/reddit/subg_uncopynode.pkl'
        with open(path, 'rb') as f:
            datas = pickle.load(f)
        return datas[0], datas[1], datas[2]

    if data == None:
        data = dataset[0]
    cluster = ClusterData(data, k)
    idx_0 = cluster.perm[cluster.partptr[0]:cluster.partptr[1]]
    idx_1 = cluster.perm[cluster.partptr[1]:cluster.partptr[2]]
    idx_2 = cluster.perm[cluster.partptr[2]:cluster.partptr[3]]
    print('cluster over!')
    # expand
    if copy_node == True:
        def expand(edge_index, idx):
            id_map = torch.zeros(max(edge_index)).bool()
            for tmp in idx:
                id_map[int(tmp)] = True
            print('id_map', len(id_map), id_map)

            extra_node = set()
            id_set = set()
            for tmp in idx:
                id_set.add(int(tmp))
            new_edge_index = [[], []]

            for i in range(len(edge_index[0])):
                if i % 1000000 == 0:
                    print(i)
                u, v = edge_index[0][i], edge_index[1][i]
                if id_map[int(u)]:
                    new_edge_index[0].append(u)
                    new_edge_index[1].append(v)
                    extra_node.add(int(v))
            # print(len(id_set))
            for id in extra_node:
                id_set.add(id)
            return torch.LongTensor(list(id_set)), torch.LongTensor(new_edge_index)
    else:
        def expand(edge_index, idx):
            id_set = torch.zeros(max(edge_index)).bool()
            for tmp in idx:
                id_set[int(tmp)] = True
            new_edge_index = [[], []]
            for i in range(len(edge_index[0])):
                if i % 1000000 == 0:
                    print(i)
                u, v = edge_index[0][i], edge_index[1][i]
                if id_set[int(u)] and id_set[int(v)]:
                    new_edge_index[0].append(u)
                    new_edge_index[1].append(v)
            return idx, torch.LongTensor(new_edge_index)

    edge_index = data.edge_index
    # print(edge_index)
    idx_0, edge_index_0 = expand(edge_index, idx_0)
    idx_1, edge_index_1 = expand(edge_index, idx_1)
    idx_2, edge_index_2 = expand(edge_index, idx_2)
    print('subgraph expands over')

    def hashing(idx, edge_index):
        idx_map = {}
        a, rank = idx.sort(0, descending=False)
        idx = a
        for i in range(len(idx)):
            idx_map[int(idx[i])] = i
        # print(idx_map)
        # print(edge_index)
        for i in range(len(edge_index[0])):
            edge_index[0][i], edge_index[1][i] = idx_map[int(
                edge_index[0][i])], idx_map[int(edge_index[1][i])]
        # print(edge_index)
        return torch.LongTensor(edge_index)
    # print(torch.max(edge_index_0))
    edge_index_0 = hashing(idx_0, edge_index_0)
    # print(torch.max(edge_index_0))
    edge_index_1 = hashing(idx_1, edge_index_1)
    edge_index_2 = hashing(idx_2, edge_index_2)

    print('hash over')
    x = data.x
    idx_0_bool_tensor = torch.zeros(len(x)).bool()
    for idx in idx_0:
        idx_0_bool_tensor[idx] = True

    idx_1_bool_tensor = torch.zeros(len(x)).bool()
    for idx in idx_1:
        idx_1_bool_tensor[idx] = True

    idx_2_bool_tensor = torch.zeros(len(x)).bool()
    for idx in idx_2:
        idx_2_bool_tensor[idx] = True

    x_0 = x[idx_0_bool_tensor]
    x_1 = x[idx_1_bool_tensor]
    x_2 = x[idx_2_bool_tensor]

    y = data.y
    y_0 = y[idx_0_bool_tensor]
    y_1 = y[idx_1_bool_tensor]
    y_2 = y[idx_2_bool_tensor]

    train_mask = data.train_mask
    train_mask_0 = train_mask[idx_0_bool_tensor]
    train_mask_1 = train_mask[idx_1_bool_tensor]
    train_mask_2 = train_mask[idx_2_bool_tensor]

    val_mask = data.val_mask
    val_mask_0 = val_mask[idx_0_bool_tensor]
    val_mask_1 = val_mask[idx_1_bool_tensor]
    val_mask_2 = val_mask[idx_2_bool_tensor]

    test_mask = data.test_mask
    test_mask_0 = test_mask[idx_0_bool_tensor]
    test_mask_1 = test_mask[idx_1_bool_tensor]
    test_mask_2 = test_mask[idx_2_bool_tensor]
    data_0 = Data(x=x_0, y=y_0, edge_index=edge_index_0,
                  train_mask=train_mask_0, val_mask=val_mask_0, test_mask=test_mask_0)
    data_1 = Data(x=x_1, y=y_1, edge_index=edge_index_1,
                  train_mask=train_mask_1, val_mask=val_mask_1, test_mask=test_mask_1)
    data_2 = Data(x=x_2, y=y_2, edge_index=edge_index_2,
                  train_mask=train_mask_2, val_mask=val_mask_2, test_mask=test_mask_2)
    print('sub graph pack over')
    return data_0, data_1, data_2


def generate_mask_from_idx(idxs, total):
    ans = torch.zeros(total).bool()
    for idx in idxs:
        ans[idx] = True
    return ans


def output_data(data, path):
    with open(path, 'wb') as f:
        pickle.dump(data, f)
    return


def read_data(path):
    with open(path, 'rb') as f:
        data = pickle.load(f)
    return data


def generate_new_data(data, train_mask, val_mask, test_mask):
    return Data(x=data.x, y=data.y, edge_index=data.edge_index, train_mask=train_mask, val_mask=val_mask, test_mask=test_mask)


if __name__ == "__main__":
    read_ipport()
    pass
