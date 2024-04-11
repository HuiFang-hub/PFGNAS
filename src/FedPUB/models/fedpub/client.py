import time
import torch
import torch.nn.functional as F

from src.FedPUB.misc.utils import *
from src.FedPUB.models.nets import *
from src.FedPUB.modules.federated import ClientModule

class Client(ClientModule):

    def __init__(self, args, w_id, g_id, sd):
        super(Client, self).__init__(args, w_id, g_id, sd)
        n_feat = self.args.model.input_shape[-1]  # 1433
        n_clss = self.args.model.num_classes  # 7
        n_dims = self.args.model.hidden
        self.l1 = self.args.federate.l1
        self.loc_l2 = self.args.federate.loc_l2

        # test = torch.tensor([1,2]).cuda(g_id)
        # print(g_id)
        # test = torch.tensor([1,2]).to(torch.device(f"cuda:{g_id}"), non_blocking=True)
        self.model = MaskedGCN(n_feat, n_dims, n_clss, self.l1, self.args).cuda(g_id)
        self.parameters = list(self.model.parameters())
        self.checkpt_path = os.path.join(self.args.results_dir,'checkpt')
        self.n_eps = self.args.federate.n_eps
        # test = self.checkpt_path

    def init_state(self):
        self.optimizer = torch.optim.Adam(self.parameters, lr=self.args.train.optimizer.lr, weight_decay=self.args.train.optimizer.weight_decay)
        self.log = {
            'lr': [],'train_lss': [],
            'ep_local_val_lss': [],'ep_local_val_acc': [],'ep_local_val_roc_auc': [],
            'rnd_local_val_lss': [],'rnd_local_val_acc': [],'rnd_local_val_roc_auc': [],
            'ep_local_test_lss': [],'ep_local_test_acc': [],'ep_local_test_roc_auc': [],
            'rnd_local_test_lss': [],'rnd_local_test_acc': [],'rnd_local_test_roc_auc': [],
            'rnd_sparsity':[], 'ep_sparsity':[]
        }

    def save_state(self):

        torch_save(self.checkpt_path, f'{self.client_id}_state.pt', {
            'optimizer': self.optimizer.state_dict(),
            'model': get_state_dict(self.model),
            'log': self.log,
        })

    def load_state(self):
        loaded = torch_load(self.checkpt_path, f'{self.client_id}_state.pt')
        set_state_dict(self.model, loaded['model'], self.gpu_id)
        self.optimizer.load_state_dict(loaded['optimizer'])
        self.log = loaded['log']
    
    def on_receive_message(self, curr_rnd):
        self.curr_rnd = curr_rnd
        self.update(self.sd[f'personalized_{self.client_id}' \
            if (f'personalized_{self.client_id}' in self.sd) else 'global'])
        self.global_w = convert_np_to_tensor(self.sd['global']['model'], self.gpu_id)

    def update(self, update):
        self.prev_w = convert_np_to_tensor(update['model'], self.gpu_id)
        set_state_dict(self.model, update['model'], self.gpu_id, skip_stat=True, skip_mask=True)

    def on_round_begin(self):
        self.train()
        self.transfer_to_server()

    def get_sparsity(self):
        n_active, n_total = 0, 1
        for mask in self.masks:
            pruned = torch.abs(mask) < self.l1
            mask = torch.ones(mask.shape).cuda(self.gpu_id).masked_fill(pruned, 0)
            n_active += torch.sum(mask)
            _n_total = 1
            for s in mask.shape:
                _n_total *= s 
            n_total += _n_total
        return ((n_total-n_active)/n_total).item()

    def train(self):
        st = time.time()
        # val_local_acc, val_local_lss = self.validate(mode='valid')
        # test_local_acc, test_local_lss = self.validate(mode='test')
        val_local_acc, val_local_lss, val_local_roc_auc = self.validate(mode='valid')
        test_local_acc, test_local_lss, test_local_roc_auc = self.validate(mode='test')
        self.logger.print(
            f'rnd: {self.curr_rnd+1}, ep: {0}, '
            f'val_local_loss: {val_local_lss.item():.4f}, val_local_acc: {val_local_acc:.4f}, val_local_roc_auc:{val_local_roc_auc:.4f}, lr: {self.get_lr()} ({time.time()-st:.2f}s)'
        )
        self.log['ep_local_val_acc'].append(val_local_acc)
        self.log['ep_local_val_lss'].append(val_local_lss)
        self.log['ep_local_val_roc_auc'].append(val_local_roc_auc)
        self.log['ep_local_test_acc'].append(test_local_acc)
        self.log['ep_local_test_lss'].append(test_local_lss)
        self.log['ep_local_test_roc_auc'].append(test_local_roc_auc)
        self.masks = []
        for name, param in self.model.state_dict().items():
            if 'mask' in name: self.masks.append(param) 

        for ep in range(self.n_eps):
            st = time.time()
            self.model.train()
            for _, batch in enumerate(self.loader.pa_loader):
                self.optimizer.zero_grad()
                batch = batch.cuda(self.gpu_id)
                y_hat = self.model(batch)
                train_lss = F.cross_entropy(y_hat[batch.train_mask], batch.y[batch.train_mask])
                
                #################################################################
                for name, param in self.model.state_dict().items():
                    if 'mask' in name:
                        train_lss += torch.norm(param.float(), 1) * self.l1
                    elif 'conv' in name or 'clsif' in name:
                        if self.curr_rnd == 0: continue
                        train_lss += torch.norm(param.float()-self.prev_w[name], 2) * self.loc_l2
                #################################################################
                        
                train_lss.backward()
                self.optimizer.step()
            
            sparsity = self.get_sparsity()
            val_local_acc, val_local_lss,val_local_roc_auc = self.validate(mode='valid')
            test_local_acc, test_local_lss,test_local_roc_auc = self.validate(mode='test')
            self.logger.print(
                f'rnd: {self.curr_rnd+1}, ep: {ep+1}, '
                f'val_local_loss: {val_local_lss.item():.4f}, test_local_acc: {test_local_acc:.4f}, test_local_roc_auc:{test_local_roc_auc:.4f}, lr: {self.get_lr()} ({time.time()-st:.2f}s)'
            )
            self.log['train_lss'].append(train_lss.item())
            self.log['ep_local_val_acc'].append(val_local_acc)
            self.log['ep_local_val_lss'].append(val_local_lss)
            self.log['ep_local_val_roc_auc'].append(val_local_roc_auc)
            self.log['ep_local_test_acc'].append(test_local_acc)
            self.log['ep_local_test_lss'].append(test_local_lss)
            self.log['ep_local_test_roc_auc'].append(test_local_roc_auc)
            self.log['ep_sparsity'].append(sparsity)
        self.log['rnd_local_val_acc'].append(val_local_acc)
        self.log['rnd_local_val_lss'].append(val_local_lss)
        self.log['rnd_local_val_roc_auc'].append(val_local_roc_auc)
        self.log['rnd_local_test_acc'].append(test_local_acc)
        self.log['rnd_local_test_lss'].append(test_local_lss)
        self.log['rnd_local_test_roc_auc'].append(test_local_roc_auc)
        self.log['rnd_sparsity'].append(sparsity)
        self.save_log()

    @torch.no_grad()
    def get_functional_embedding(self):
        self.model.eval()
        with torch.no_grad():
            proxy_in = self.sd['proxy']
            proxy_in = proxy_in.cuda(self.gpu_id)
            proxy_out = self.model(proxy_in, is_proxy=True)
            proxy_out = proxy_out.mean(dim=0)
            proxy_out = proxy_out.clone().detach().cpu().numpy()
        return proxy_out

    def transfer_to_server(self):
        self.sd[self.client_id] = {
            'model': get_state_dict(self.model),
            'train_size': len(self.loader.partition),
            'functional_embedding': self.get_functional_embedding()
        }




    
    
