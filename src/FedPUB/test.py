
import torch.optim as optim
import sys
import time
import atexit

from datetime import datetime

from models.nets import *
import os
from src.federatedscope.core.cmd_args import parse_args, parse_client_cfg
from src.federatedscope.core.configs.config import global_cfg
# from src.FedPUB.modules.multiprocs import ParentProcess
from src.federatedscope.core.auxiliaries.logging import update_logger,get_resfile_path
import torch.multiprocessing as mp
import torch
from parser import Parser



import os
import torch


# from torch_geometric.transforms import BaseTransform
# from torch_geometric.utils import to_scipy_sparse_matrix


# 定义模型
class SimpleModel(nn.Module):
    def __init__(self):
        super(SimpleModel, self).__init__()
        self.fc = nn.Linear(10, 5)

    def forward(self, x):
        return self.fc(x)


def train(args,rank):
    # 设置每个进程的设备
    # torch.cuda.set_device(rank)

    # 创建模型并将其放在每个GPU上
    model = SimpleModel().cuda(rank)

    # 定义损失函数和优化器
    criterion = nn.MSELoss()
    optimizer = optim.SGD(model.parameters(), lr= 0.001)

    # 模拟一些输入数据
    input_data = torch.randn(5, 10).cuda(rank)
    target = torch.randn(5, 5).cuda(rank)

    for epoch in range(10):
        # 向模型输入数据并计算损失
        output = model(input_data)
        loss = criterion(output, target)

        # 反向传播和优化
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        print(f'Rank {rank}, Epoch {epoch + 1}/{10}, Loss: {loss.item()}')

def set_config(args):

    args.base_lr = 1e-3
    args.min_lr = 1e-3
    args.momentum_opt = 0.9
    args.weight_decay = 1e-6
    args.warmup_epochs = 10
    args.base_momentum = 0.99
    args.final_momentum = 1.0

    args.model = 'fedpub'
    args.mode = 'disjoint'
    args.dataset = 'Cora'
    args.frac = 1.0
    args.n_workers = 4


    if args.dataset == 'Cora':
        args.n_feat = 1433  #?
        args.n_clss = 7  #?
        args.n_clients = 3 if args.n_clients == None else args.n_clients
        args.base_lr = 0.01 if args.lr == None else args.lr

    now = datetime.now().strftime("%Y%m%d_%H%M%S")
    trial = f'{args.dataset}_{args.mode}/clients_{args.n_clients}/{now}_{args.model}'

    args.data_path = f'datasets'
    args.checkpt_path = f'{args.base_path}/checkpoints/{trial}'
    args.log_path = f'{args.base_path}/logs/{trial}'

    if args.debug == True:
        args.checkpt_path = f'{args.base_path}/debug/checkpoints/{trial}'
        args.log_path = f'{args.base_path}/debug/logs/{trial}'

    return args


class ParentProcess:
    def __init__(self, args, Server, Client):
        self.args = args
        self.gpus = [2,3,4] #[int(g) for g in args.gpu.split(',')]
        self.gpu_server = 2
        self.proc_id = os.getppid()
        print(f'main process id: {self.proc_id}')
        self.n_workers = 3
        self.sd = mp.Manager().dict()
        self.sd['is_done'] = False
        self.create_workers(Client,args)

        self.server = Server(args, self.sd, self.gpu_server)
        atexit.register(self.done)

    def create_workers(self, Client,args):
        self.processes = []
        self.q = {}
        for worker_id in range(self.n_workers):
            # gpu_id = self.gpus[worker_id] if worker_id <= len(self.gpus)-1 else self.gpus[worker_id%len(self.gpus)]
            gpu_id = self.gpus[worker_id+1] if worker_id < len(self.gpus)-1 else self.gpus[(worker_id-(len(self.gpus)-1))%len(self.gpus)]
            print(f'worker_id: {worker_id}, gpu_id:{gpu_id}')
            self.q[worker_id] = mp.Queue()
            # p = mp.Process(target=WorkerProcess, args=(args, worker_id, gpu_id, self.q[worker_id], self.sd, Client))
            p = mp.Process(target=train, args=(args, gpu_id))
            p.start()
            self.processes.append(p)
        print('okkkkkkkkkkkkkkkkkkkkkkkkkkkkkk!')

    def start(self):
        self.sd['is_done'] = False
        if os.path.isdir(self.args.checkpt_path) == False:
            os.makedirs(self.args.checkpt_path)
        if os.path.isdir(self.args.log_path) == False:
            os.makedirs(self.args.log_path)
        self.n_connected = round(self.args.n_clients*self.args.frac)
        for curr_rnd in range(10):
            self.curr_rnd = curr_rnd
            self.updated = set()
            np.random.seed(self.args.seed+curr_rnd)
            self.selected = np.random.choice(self.args.n_clients, self.n_connected, replace=False).tolist()
            st = time.time()
            ##################################################
            self.server.on_round_begin(curr_rnd)
            ##################################################
            while len(self.selected)>0:
                _selected = []
                for worker_id, q in self.q.items():
                    c_id = self.selected.pop(0)
                    _selected.append(c_id)
                    q.put((c_id, curr_rnd))
                    if len(self.selected) == 0:
                        break
                self.wait(curr_rnd, _selected)
            # print(f'[main] all clients updated at round {curr_rnd}')
            ###########################################
            self.server.on_round_complete(self.updated)
            ###########################################
            print(f'[main] round {curr_rnd} done ({time.time()-st:.2f} s)')

        self.sd['is_done'] = True
        for worker_id, q in self.q.items():
            q.put(None)
        print('[main] server done')
        sys.exit()

    def wait(self, curr_rnd, _selected):
        cont = True
        while cont:
            cont = False
            for c_id in _selected:
                if not c_id in self.sd:
                    cont = True
                else:
                    self.updated.add(c_id)
            time.sleep(0.1)

    def done(self):
        for p in self.processes:
            p.join()
        print('[main] All children have joined. Destroying main process ...')

def generate_data_partition(init_cfg):
    if init_cfg.data.mode == 'disjoint':
        from src.FedPUB.data.generators.disjoint import generate_data
        # for n_clients in init_cfg.federate.client_num:
        config = generate_data(init_cfg)
    # else:
    #     from src.FedPUB.data.generators.overlapping import generate_data
    #     comms = [2, 5]
    #     for n_comms in comms:
    #         generate_data(dataset=init_cfg.data.type, n_comms=n_comms)
    return config

if __name__ == '__main__':
    # mp.set_start_method('spawn')


    # init_cfg = global_cfg.clone()
    # args = parse_args()
    # if args.cfg_file:
    #     init_cfg.merge_from_file(args.cfg_file)
    # cfg_opt, client_cfg_opt = parse_client_cfg(args.opts)
    # init_cfg.merge_from_list(cfg_opt)
    # init_cfg.federate.n_workers = init_cfg.federate.client_num + 1
    # 然后继续你的主程序
    # _,modified_cfg = get_data(init_cfg)
    # init_cfg.merge_from_other_cfg(modified_cfg)
    # fl
    # if init_cfg.federate.method == 'FedPUB':
    #     from src.FedPUB.models.fedpub.server import Server
    #     from src.FedPUB.models.fedpub.client import Client
    #
    # elif init_cfg.federate.method == 'fedavg':
    #     from src.FedPUB.models.fedavg.server import Server
    #     from src.FedPUB.models.fedavg.client import Client
    #
    # else:
    #     print('incorrect model was given: {}'.format(init_cfg.federate.method))
    #     os._exit(0)
    #
    # pp = ParentProcess(init_cfg, Server, Client)
    # pp.start()

    args = set_config(Parser().parse())
    # 然后继续你的主程序

    if args.model == 'fedpub':
        from models.fedpub.server import Server
        from models.fedpub.client import Client
    elif args.model == 'fedavg':
        from models.fedavg.server import Server
        from models.fedavg.client import Client

    else:
        print('incorrect model was given: {}'.format(args.model))
        os._exit(0)
    # pp = ParentProcess(init_cfg, Server, Client)
    pp = ParentProcess(args, Server, Client)
    pp.start()
