import torch
import copy
import numpy as np
import torch.nn.functional as F

from src.federatedscope.gfl.loss import GreedyLoss
from src.federatedscope.gfl.trainer.nodetrainer import NodeFullBatchTrainer


def gad_loss_function( logits, batch_size,negsamp_ratio=1):
    b_xent = torch.nn.BCEWithLogitsLoss(reduction='none',
                                        pos_weight=torch.tensor(
                                            [negsamp_ratio]))

    lbl = torch.unsqueeze(
        torch.cat((torch.ones(batch_size),
                   torch.zeros(batch_size * negsamp_ratio))), 1)

    score = b_xent(logits.cpu(), lbl.cpu())

    return score
class LocalGenTrainer(NodeFullBatchTrainer):
    def __init__(self,
                 model,
                 data,
                 device,
                 config,
                 only_for_eval=False,
                 monitor=None):
        super(LocalGenTrainer, self).__init__(model, data, device, config,
                                              only_for_eval, monitor)
        self.criterion_num = F.smooth_l1_loss
        self.criterion_feat = GreedyLoss

    def _hook_on_batch_forward(self, ctx):
        batch = ctx.data_batch.to(ctx.device)
        mask = batch['{}_mask'.format(ctx.cur_mode)]
        pred_missing, pred_feat, nc_pred = ctx.model(batch)
        pred_missing, pred_feat, nc_pred = pred_missing[mask], pred_feat[
            mask], nc_pred[mask]
        loss_num = self.criterion_num(pred_missing, batch.num_missing[mask])
        loss_feat = self.criterion_feat(
            pred_feats=pred_feat,
            true_feats=batch.x_missing[mask],
            pred_missing=pred_missing,
            true_missing=batch.num_missing[mask],
            num_pred=self.cfg.fedsageplus.num_pred).requires_grad_()
        loss_clf = ctx.criterion(nc_pred, batch.y[mask])
        ctx.batch_size = torch.sum(mask).item()
        ctx.loss_batch = (self.cfg.fedsageplus.a * loss_num +
                          self.cfg.fedsageplus.b * loss_feat +
                          self.cfg.fedsageplus.c * loss_clf).float()

        ctx.y_true = batch.num_missing[mask]
        ctx.y_prob = pred_missing

    def _hook_on_fit_end(self, ctx):
        """
        Evaluate metrics.

        Note:
          The modified attributes and according operations are shown below:
            ==================================  ===========================
            Attribute                           Operation
            ==================================  ===========================
            ``ctx.ys_true``                     Convert to ``numpy.array``
            ``ctx.ys_prob``                     Convert to ``numpy.array``
            ``ctx.monitor``                     Evaluate the results
            ``ctx.eval_metrics``                Get evaluated results from \
            ``ctx.monitor``
            ==================================  ===========================
        """
        # ctx.ys_true = CtxVar(torch.cat(ctx.ys_true), LIFECYCLE.ROUTINE)
        # ctx.ys_prob = CtxVar(torch.cat(ctx.ys_prob), LIFECYCLE.ROUTINE)
        # results = ctx.monitor.eval(ctx)
        results = {}
        setattr(ctx, 'eval_metrics', results)
        # pass


class LocalGenTrainer_dgl(NodeFullBatchTrainer):
    # 去除clf
    def __init__(self,
                 model,
                 data,
                 device,
                 config,
                 only_for_eval=False,
                 monitor=None):
        super(LocalGenTrainer_dgl, self).__init__(model, data, device, config,
                                              only_for_eval, monitor)
        self.criterion_num = F.smooth_l1_loss
        self.criterion_feat = GreedyLoss

    def _hook_on_batch_forward(self, ctx):
        batch = ctx.data_batch.to(ctx.device)
        mask = batch['{}_mask'.format(ctx.cur_mode)]
        pred_missing, pred_feat, _ = ctx.model(batch)
        pred_missing, pred_feat = pred_missing[mask], pred_feat[mask]
        loss_num = self.criterion_num(pred_missing, batch.num_missing[mask])
        loss_feat = self.criterion_feat(
            pred_feats=pred_feat,
            true_feats=batch.x_missing[mask],
            pred_missing=pred_missing,
            true_missing=batch.num_missing[mask],
            num_pred=self.cfg.fedsageplus.num_pred).requires_grad_()
        # loss_clf = ctx.criterion(nc_pred, batch.y[mask])
        ctx.batch_size = torch.sum(mask).item()
        ctx.loss_batch = (self.cfg.fedsageplus.a * loss_num +
                          self.cfg.fedsageplus.b * loss_feat).float()

        ctx.y_true = batch.num_missing[mask]
        ctx.y_prob = pred_missing


class FedGenTrainer(LocalGenTrainer):
    def _hook_on_batch_forward(self, ctx):
        batch = ctx.data_batch.to(ctx.device)
        mask = batch['{}_mask'.format(ctx.cur_mode)]
        pred_missing, pred_feat, nc_pred = ctx.model(batch)
        pred_missing, pred_feat, nc_pred = pred_missing[mask], pred_feat[
            mask], nc_pred[mask]
        loss_num = self.criterion_num(pred_missing, batch.num_missing[mask])
        loss_feat = self.criterion_feat(pred_feats=pred_feat,
                                        true_feats=batch.x_missing[mask],
                                        pred_missing=pred_missing,
                                        true_missing=batch.num_missing[mask],
                                        num_pred=self.cfg.fedsageplus.num_pred)
        loss_clf = ctx.criterion(nc_pred, batch.y[mask])
        ctx.batch_size = torch.sum(mask).item()
        ctx.loss_batch = (self.cfg.fedsageplus.a * loss_num +
                          self.cfg.fedsageplus.b * loss_feat +
                          self.cfg.fedsageplus.c *
                          loss_clf).float() / self.cfg.federate.client_num

        ctx.y_true = batch.num_missing[mask]
        ctx.y_prob = pred_missing

    def update_by_grad(self, grads):
        """
        Arguments:
            grads: grads of other clients to optimize the local model
        :returns:
            state_dict of generation model
        """
        for key in grads.keys():
            if isinstance(grads[key], list):
                grads[key] = torch.FloatTensor(grads[key]).to(self.ctx.device)

        for key, value in self.ctx.model.named_parameters():
            value.grad += grads[key].to(self.ctx.cfg.device)
        self.ctx.optimizer.step()
        return self.ctx.model.cpu().state_dict()

    def cal_grad(self, raw_data, model_para, embedding, true_missing):
        """
        Arguments:
            raw_data (Pyg.Data): raw graph
            model_para: model parameters
            embedding: output embeddings after local encoder
            true_missing: number of missing node
        :returns:
            grads: grads to optimize the model of other clients
        """
        para_backup = copy.deepcopy(self.ctx.model.cpu().state_dict())

        for key in model_para.keys():
            if isinstance(model_para[key], list):
                model_para[key] = torch.FloatTensor(model_para[key])
        self.ctx.model.load_state_dict(model_para)
        self.ctx.model = self.ctx.model.to(self.ctx.device)
        self.ctx.model.train()

        raw_data = raw_data.to(self.ctx.device)
        embedding = torch.FloatTensor(embedding).to(self.ctx.device)
        true_missing = true_missing.long().to(self.ctx.device)
        pred_missing = self.ctx.model.reg_model(embedding)
        pred_feat = self.ctx.model.gen(embedding)

        # Random pick node and compare its neighbors with predicted nodes
        choice = np.random.choice(raw_data.num_nodes, embedding.shape[0])
        global_target_feat = []
        for c_i in choice:
            neighbors_ids = raw_data.edge_index[1][torch.where(
                raw_data.edge_index[0] == c_i)[0]]
            while len(neighbors_ids) == 0:
                id_i = np.random.choice(raw_data.num_nodes, 1)[0]
                neighbors_ids = raw_data.edge_index[1][torch.where(
                    raw_data.edge_index[0] == id_i)[0]]
            choice_i = np.random.choice(neighbors_ids.detach().cpu().numpy(),
                                        self.cfg.fedsageplus.num_pred)
            for ch_i in choice_i:
                global_target_feat.append(
                    raw_data.x[ch_i].detach().cpu().numpy())
        global_target_feat = np.asarray(global_target_feat).reshape(
            (embedding.shape[0], self.cfg.fedsageplus.num_pred,
             raw_data.num_node_features))
        loss_feat = self.criterion_feat(pred_feats=pred_feat,
                                        true_feats=global_target_feat,
                                        pred_missing=pred_missing,
                                        true_missing=true_missing,
                                        num_pred=self.cfg.fedsageplus.num_pred)
        loss = self.cfg.fedsageplus.b * loss_feat
        loss = (1.0 / self.cfg.federate.client_num * loss).requires_grad_()
        loss.backward()
        grads = {
            key: value.grad
            for key, value in self.ctx.model.named_parameters()
        }
        # Rollback
        self.ctx.model.load_state_dict(para_backup)
        return grads

    @torch.no_grad()
    def embedding(self):
        model = self.ctx.model.to(self.ctx.device)
        data = self.ctx.data['data'].to(self.ctx.device)
        return model.encoder_model(data.x,data.edge_index).to('cpu')

