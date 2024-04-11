import time
import numpy as np

from scipy.spatial.distance import cosine

from src.FedPUB.misc.utils import *
from src.FedPUB.models.nets import *
from src.FedPUB.modules.federated import ServerModule

# class Server(ServerModule):
#     def __init__(self, args, sd, gpu_server):
#         super(Server, self).__init__(args, sd, gpu_server)
#         n_feat = self.args.model.input_shape[-1] # 1433
#         n_clss = self.args.model.num_classes #7
#         n_dims = self.args.model.hidden
#         l1 = self.args.federate.l1
#         # n_feat = 1433
#         # n_clss =  7
#         # n_dims = 64
#         # l1 = 1e-3
#         # test = torch.tensor([1,2]).cuda(self.gpu_id)
#         self.model = MaskedGCN(n_feat, n_dims, n_clss, l1, self.args).cuda(self.gpu_id)
#         self.sd['proxy'] = self.get_proxy_data(n_feat)
#         self.update_lists = []
#         self.sim_matrices = []
#         self.checkpt_path = os.path.join(self.args.results_dir, 'checkpt')
#         # print(self.checkpt_path)
#         # test = self.checkpt_path
#
#     def get_proxy_data(self, n_feat):
#         import networkx as nx
#
#         num_graphs, num_nodes = self.args.federate.n_proxy, 100
#         data = from_networkx(nx.random_partition_graph([num_nodes] * num_graphs, p_in=0.1, p_out=0, seed=self.args.seed))
#         data.x = torch.normal(mean=0, std=1, size=(num_nodes * num_graphs, n_feat))
#         return data
#
#     def on_round_begin(self, curr_rnd):
#         self.round_begin = time.time()
#         self.curr_rnd = curr_rnd
#         self.sd['global'] = self.get_weights()
#
#     def on_round_complete(self, updated):
#         self.update(updated)
#         self.save_state()
#
#     def update(self, updated):
#         st = time.time()
#         local_weights = []
#         local_functional_embeddings = []
#         local_train_sizes = []
#         for c_id in updated:
#             local_weights.append(self.sd[c_id]['model'].copy())
#             local_functional_embeddings.append(self.sd[c_id]['functional_embedding'])
#             local_train_sizes.append(self.sd[c_id]['train_size'])
#             del self.sd[c_id]
#         self.logger.print(f'all clients have been uploaded ({time.time()-st:.2f}s)')
#
#         n_connected = round(self.args.federate.client_num*self.args.federate.frac)
#         assert n_connected == len(local_functional_embeddings)
#         sim_matrix = np.empty(shape=(n_connected, n_connected))
#         for i in range(n_connected):
#             for j in range(n_connected):
#                 sim_matrix[i, j] = 1 - cosine(local_functional_embeddings[i], local_functional_embeddings[j])
#
#         if self.args.federate.agg_norm == 'exp':
#             sim_matrix = np.exp(self.args.federate.norm_scale * sim_matrix)
#
#         row_sums = sim_matrix.sum(axis=1)
#         sim_matrix = sim_matrix / row_sums[:, np.newaxis]
#
#         st = time.time()
#         ratio = (np.array(local_train_sizes)/np.sum(local_train_sizes)).tolist()
#         self.set_weights(self.model, self.aggregate(local_weights, ratio))
#         self.logger.print(f'global model has been updated ({time.time()-st:.2f}s)')
#
#         st = time.time()
#         for i, c_id in enumerate(updated):
#             aggr_model_weights = self.aggregate(local_weights, sim_matrix[i, :])
#             if f'personalized_{c_id}' in self.sd: del self.sd[f'personalized_{c_id}']
#             self.sd[f'personalized_{c_id}'] = {'model': aggr_local_model_weights}
#         self.update_lists.append(updated)
#         self.sim_matrices.append(sim_matrix)
#         self.logger.print(f'local model has been updated ({time.time()-st:.2f}s)')
#
#     def set_weights(self, model, state_dict):
#         set_state_dict(model, state_dict, self.gpu_id)
#
#     def get_weights(self):
#         return {
#             'model': get_state_dict(self.model),
#         }
#
#     def save_state(self):
#         torch_save(self.checkpt_path, 'server_state.pt', {
#             'model': get_state_dict(self.model),
#             'sim_matrices': self.sim_matrices,
#             'update_lists': self.update_lists
#         })




class Server(ServerModule):
    def __init__(self, args, sd, gpu_server):
        super(Server, self).__init__(args, sd, gpu_server)
        n_feat = self.args.model.input_shape[-1]  # 1433
        n_clss = self.args.model.num_classes  # 7
        n_dims = self.args.model.hidden
        l1 = self.args.federate.l1
        self.model = MaskedGCN(n_feat, n_dims, n_clss, l1, self.args).cuda(self.gpu_id)
        self.parameters = list(self.model.parameters())
        self.sd['proxy'] = self.get_proxy_data(n_feat)
        self.update_lists = []
        self.sim_matrices = []
        self.checkpt_path = os.path.join(self.args.results_dir, 'checkpt')
        # print(self.checkpt_path)
        # test = self.checkpt_path
        self.log = {
            # 'ep_val_lss': [], 'ep_val_acc': [], 'ep_val_roc_auc': [],
            # 'rnd_val_lss': [], 'rnd_val_acc': [], 'rnd_val_roc_auc': [],
            # 'ep_test_lss': [], 'ep_test_acc': [], 'ep_test_roc_auc': [],
            'rnd_test_lss': [], 'rnd_test_acc': [], 'rnd_test_roc_auc': [],

        }

    def get_proxy_data(self, n_feat):
        import networkx as nx

        num_graphs, num_nodes = self.args.federate.n_proxy, 100
        data = from_networkx(
            nx.random_partition_graph([num_nodes] * num_graphs, p_in=0.1, p_out=0, seed=self.args.seed))
        data.x = torch.normal(mean=0, std=1, size=(num_nodes * num_graphs, n_feat))
        return data

    def on_round_begin(self, curr_rnd):
        self.round_begin = time.time()
        self.curr_rnd = curr_rnd
        self.sd['global'] = self.get_weights()

    def on_round_complete(self, updated):
        self.update(updated)
        self.save_state()

    def update(self, updated):
        st = time.time()
        local_weights = []
        local_functional_embeddings = []
        local_train_sizes = []
        for c_id in updated:
            local_weights.append(self.sd[c_id]['model'].copy())
            local_functional_embeddings.append(self.sd[c_id]['functional_embedding'])
            local_train_sizes.append(self.sd[c_id]['train_size'])
            del self.sd[c_id]
        self.logger.print(f'all clients have been uploaded ({time.time() - st:.2f}s)')

        n_connected = round(self.args.federate.client_num * self.args.federate.frac)
        assert n_connected == len(local_functional_embeddings)
        sim_matrix = np.empty(shape=(n_connected, n_connected))
        for i in range(n_connected):
            for j in range(n_connected):
                sim_matrix[i, j] = 1 - cosine(local_functional_embeddings[i], local_functional_embeddings[j])

        if self.args.federate.agg_norm == 'exp':
            sim_matrix = np.exp(self.args.federate.norm_scale * sim_matrix)

        row_sums = sim_matrix.sum(axis=1)
        sim_matrix = sim_matrix / row_sums[:, np.newaxis]

        st = time.time()
        ratio = (np.array(local_train_sizes) / np.sum(local_train_sizes)).tolist()
        self.set_weights(self.model, self.aggregate(local_weights, ratio))
        self.logger.print(f'global model has been updated ({time.time() - st:.2f}s)')

        self.global_model_test()

        st = time.time()
        for i, c_id in enumerate(updated):
            aggr_local_model_weights = self.aggregate(local_weights, sim_matrix[i, :])
            if f'personalized_{c_id}' in self.sd: del self.sd[f'personalized_{c_id}']
            self.sd[f'personalized_{c_id}'] = {'model': aggr_local_model_weights}
        self.update_lists.append(updated)
        self.sim_matrices.append(sim_matrix)
        self.logger.print(f'local model has been updated ({time.time() - st:.2f}s)')

    def set_weights(self, model, state_dict):
        set_state_dict(model, state_dict, self.gpu_id)

    def get_weights(self):
        return {
            'model': get_state_dict(self.model),
        }

    def save_state(self):
        torch_save(self.checkpt_path, 'server_state.pt', {
            'model': get_state_dict(self.model),
            'sim_matrices': self.sim_matrices,
            'update_lists': self.update_lists
        })

    def global_model_test(self):

        st = time.time()

        # val_acc, val_lss, val_roc_auc = self.validate(mode='valid')
        # test_acc, test_lss, test_roc_auc = self.validate(mode='test')
        # self.logger.print(
        #     f'rnd: {self.curr_rnd + 1}, ep: {0}, '
        #     f'val_loss: {val_lss.item():.4f}, val_acc: {val_acc:.4f}, val_roc_auc:{val_roc_auc:.4f}, lr: {self.get_lr()} ({time.time() - st:.2f}s)'
        # )
        # self.log['ep_val_acc'].append(val_acc)
        # self.log['ep_val_lss'].append(val_lss)
        # self.log['ep_val_roc_auc'].append(val_roc_auc)
        # self.log['ep_test_acc'].append(test_acc)
        # self.log['ep_test_lss'].append(test_lss)
        # self.log['ep_test_roc_auc'].append(test_roc_auc)
        # self.masks = []
        # for name, param in self.model.state_dict().items():
        #     if 'mask' in name: self.masks.append(param)

        # for ep in range(self.n_eps):
        #     st = time.time()
        self.model.eval()
        # for _, batch in enumerate(self.loader.pa_loader):
        #     self.optimizer.zero_grad()
        #     batch = batch.cuda(self.gpu_id)
        #     y_hat = self.model(batch)
        #     train_lss = F.cross_entropy(y_hat[batch.train_mask], batch.y[batch.train_mask])
        #
        #     #################################################################
        #     for name, param in self.model.state_dict().items():
        #         if 'mask' in name:
        #             train_lss += torch.norm(param.float(), 1) * self.l1
        #         elif 'conv' in name or 'clsif' in name:
        #             if self.curr_rnd == 0: continue
        #             train_lss += torch.norm(param.float() - self.prev_w[name], 2) * self.loc_l2
        #     #################################################################
        #
        #     train_lss.backward()
        #     self.optimizer.step()

        # sparsity = self.get_sparsity()
        # val_acc, val_lss, val_roc_auc = self.validate(mode='valid')
        test_acc, test_lss, test_roc_auc = self.validate(mode='test')
        self.logger.print(
            f'test_acc: {test_acc:.4f}, test_roc_auc:{test_roc_auc:.4f}'
        )
        # self.log['ep_val_acc'].append(val_acc)
        # self.log['ep_val_lss'].append(val_lss)
        # self.log['ep_val_roc_auc'].append(val_roc_auc)
        # self.log['ep_test_acc'].append(test_acc)
        # self.log['ep_test_lss'].append(test_lss)
        # self.log['ep_test_roc_auc'].append(test_roc_auc)
        # self.log['ep_sparsity'].append(sparsity)
        # self.log['rnd_val_acc'].append(val_acc)
        # self.log['rnd_val_lss'].append(val_lss)
        # self.log['rnd_val_roc_auc'].append(val_roc_auc)
        self.log['rnd_test_acc'].append(test_acc)
        self.log['rnd_test_lss'].append(test_lss)
        self.log['rnd_test_roc_auc'].append(test_roc_auc)
        # self.log['rnd_sparsity'].append(sparsity)
        self.save_log()




    def save_log(self):
        save(self.args.results_dir, 'global.txt', {
                # 'args': self.args,
                'log': self.log
            })



