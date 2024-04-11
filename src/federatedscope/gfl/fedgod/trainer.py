import copy
import numpy as np
import random
import math
from numpy import percentile
from scipy.special import erf
from scipy.stats import binom
from sklearn.preprocessing import MinMaxScaler
from sklearn.utils.validation import check_is_fitted
from sklearn.utils.multiclass import check_classification_targets
import torch.nn.functional as F
from src.federatedscope.core.auxiliaries.dataloader_builder import get_dataloader
from src.federatedscope.core.auxiliaries.ReIterator import ReIterator
from src.federatedscope.core.data.wrap_dataset import WrapDataset
from src.federatedscope.core.trainers import GeneralTorchTrainer
from src.federatedscope.core.trainers.context import CtxVar
from src.federatedscope.core.trainers.enums import MODE, LIFECYCLE
from src.federatedscope.gfl.fedgod.utils import *
from src.federatedscope.gfl.fedgod.metrics import *
from src.federatedscope.core.auxiliaries.optimizer_builder import get_optimizer
from src.federatedscope.core.auxiliaries.scheduler_builder import get_scheduler
class fedgod_trainer(GeneralTorchTrainer):
    def register_default_hooks_train(self):
        super(fedgod_trainer, self).register_default_hooks_train()
        # self.register_hook_in_train(self._hook_on_fit_start_init,
        #                             "on_fit_start")
        # self.register_hook_in_train(
        #     self._hook_on_fit_start_calculate_model_size, "on_fit_start")
        # self.register_hook_in_train(self._hook_on_epoch_start,
        #                             "on_epoch_start")
        # self.register_hook_in_train(self._hook_on_batch_start_init,
        #                             "on_batch_start")
        # self.register_hook_in_train(self._hook_on_batch_forward,
        #                             "on_batch_forward")
        # self.register_hook_in_train(self._hook_on_batch_forward_regularizer,
        #                             "on_batch_forward")
        # self.register_hook_in_train(self._hook_on_batch_forward_flop_count,
        #                             "on_batch_forward")
        # self.register_hook_in_train(self._hook_on_batch_backward,
        #                             "on_batch_backward")
        # self.register_hook_in_train(self._hook_on_batch_end,
        #                             "on_batch_end")
        self.register_hook_in_train(self._hook_on_epoch_end,
                                    "on_epoch_end",insert_pos=-1)
        # self.register_hook_in_train(self._hook_on_fit_end,
        #                             "on_fit_end")
        # self.register_hook_in_train(self._hook_on_predict,
        #                             "on_fit_end",insert_pos=-1)

        # self.register_hook_in_train(self._hook_on_decision_function,
        #                             "on_fit_end", insert_pos=-1)
    # def register_default_hooks_eval(self):
    #     super(fedgod_trainer, self).register_default_hooks_eval()
    #     # test/val
    #     self.register_hook_in_eval(self._hook_on_fit_start_init,
    #                                "on_fit_start")
    #     self.register_hook_in_eval(self._hook_on_epoch_start, "on_epoch_start")
    #     self.register_hook_in_eval(self._hook_on_batch_start_init,
    #                                "on_batch_start")
    #     self.register_hook_in_eval(self._hook_on_batch_forward,
    #                                "on_batch_forward")
    #     self.register_hook_in_eval(self._hook_on_batch_end, "on_batch_end")
    #     self.register_hook_in_eval(self._hook_on_predict, "on_fit_end",insert_pos=-1)
    def register_default_hooks_eval(self):
        # test/val
        # self.register_hook_in_eval(self._hook_on_fit_start_init,
        #                            "on_fit_start")
        # self.register_hook_in_eval(self._hook_on_epoch_start, "on_epoch_start")
        # self.register_hook_in_eval(self._hook_on_batch_start_init,
        #                            "on_batch_start")
        # self.register_hook_in_eval(self._hook_on_batch_forward,
        #                            "on_batch_forward")
        # self.register_hook_in_eval(self._hook_on_batch_end, "on_batch_end")
        self.register_hook_in_eval(self._hook_on_predict, "on_fit_end")

    def parse_data(self, data):
        """Populate "{}_data", "{}_loader" and "num_{}_data" for different
        modes
        """
        init_dict = dict()
        if isinstance(data, dict):
            for mode in ["train", "val", "test"]:
                init_dict["{}_loader".format(mode)] = data.get(mode)
                init_dict["{}_data".format(mode)] = None
                # For node-level task dataloader contains one graph
                init_dict["num_{}_data".format(mode)] = 1
        else:
            raise TypeError("Type of data should be dict.")
        return init_dict

    def _hook_on_fit_start_init(self, ctx):
        """
        Note:
          The modified attributes and according operations are shown below:
            ==================================  ===========================
            Attribute                           Operation
            ==================================  ===========================
            ``ctx.model``                       Move to ``ctx.device``
            ``ctx.optimizer``                   Initialize by ``ctx.cfg``
            ``ctx.scheduler``                   Initialize by ``ctx.cfg``
            ``ctx.loss_batch_total``            Initialize to 0
            ``ctx.loss_regular_total``          Initialize to 0
            ``ctx.num_samples``                 Initialize to 0
            ``ctx.ys_true``                     Initialize to ``[]``
            ``ctx.ys_prob``                     Initialize to ``[]``
            ==================================  ===========================
        """
        # prepare model and optimizer

        ctx.loss_epoch_total = CtxVar(0, LIFECYCLE.ROUTINE)
        ctx.num_samples = CtxVar(0, LIFECYCLE.ROUTINE)
        ctx.model.to(ctx.device)
        ctx.model.train()
        if ctx.cur_mode == 'train':
            ctx.optimizer = get_optimizer(ctx.model,
                                          **ctx.cfg[ctx.cur_mode].optimizer)
        # ctx.scheduler = get_scheduler(ctx.optimizer,
        #                               **ctx.cfg[ctx.cur_mode].scheduler)

        # ctx.optimizer = torch.optim.Adam(ctx.model.parameters(),
        #                              lr=ctx.cfg.model.lr,
        #                              weight_decay=ctx.cfg.model.weight_decay)
        ctx.scheduler = None
        data = ctx.data['data']
        ctx.y_true = data.ay
        num_epoch = getattr(ctx, f"num_{ctx.cur_split}_epoch")
        ctx.num_epoch = CtxVar(num_epoch, LIFECYCLE.ROUTINE)
        # loader = get_dataloader(ctx.data['data'],self.cfg, ctx.cur_split)
        num_nodes, feat_dim, x, adj = process_graph(data, ctx.device)
        ctx.num_nodes = CtxVar(num_nodes, LIFECYCLE.ROUTINE)

        ctx.feat_dim = CtxVar(feat_dim, LIFECYCLE.EPOCH)
        ctx.x = CtxVar(x, LIFECYCLE.ROUTINE)
        ctx.adj = CtxVar(adj, LIFECYCLE.ROUTINE)
        if ctx.cfg.dataloader.batch_size:
            batch_num = math.ceil(num_nodes/ ctx.cfg.dataloader.batch_size)
        else:  # full batch training
            batch_num = 1


        ctx.batch_num = CtxVar(batch_num, LIFECYCLE.ROUTINE)

        multi_epoch_ano_score = np.zeros((ctx.num_epoch, num_nodes))
        ctx.multi_epoch_ano_score = CtxVar(multi_epoch_ano_score, LIFECYCLE.ROUTINE)


    def _hook_on_epoch_start(self, ctx):
        """
        Note:
          The modified attributes and according operations are shown below:
            ==================================  ===========================
            Attribute                           Operation
            ==================================  ===========================
            ``ctx.{ctx.cur_split}_loader``      Initialize DataLoader
            ==================================  ===========================
        """
        # prepare dataloader
        if ctx.get("{}_loader".format(ctx.cur_split)) is None:
            if ctx.get("{}_data".format(ctx.cur_split)) is None:
                ctx.num_samples_batch = CtxVar(0, LIFECYCLE.EPOCH)
                ctx.loss_batch_total =  CtxVar(0, LIFECYCLE.EPOCH)
                ctx.loss_regular_total =  CtxVar(0, LIFECYCLE.EPOCH)
                # ctx.epoch_loss = CtxVar(0, LIFECYCLE.EPOCH)
                all_idx = list(range(ctx.num_nodes))
                random.shuffle(all_idx)
                ctx.all_idx = CtxVar(all_idx, LIFECYCLE.EPOCH)
                decision_scores = np.zeros(ctx.num_nodes)
                ctx.decision_scores = CtxVar(decision_scores, LIFECYCLE.EPOCH)
                subgraphs = generate_rw_subgraph(ctx.data['data'], ctx.num_nodes, ctx.cfg.model.subgraph_size)
                ctx.subgraphs = CtxVar(subgraphs, LIFECYCLE.EPOCH)
            else:
                loader = get_dataloader(
                    WrapDataset(ctx.get("{}_data".format(ctx.cur_split))),
                    self.cfg, ctx.cur_split)
            # setattr(ctx, "{}_loader".format(ctx.cur_split), ReIterator(loader))
        elif not isinstance(ctx.get("{}_loader".format(ctx.cur_split)),
                            ReIterator):
            setattr(ctx, "{}_loader".format(ctx.cur_split),
                    ReIterator(ctx.get("{}_loader".format(ctx.cur_split))))
        else:
            ctx.get("{}_loader".format(ctx.cur_split)).reset()

    def _hook_on_batch_start_init(self, ctx):
        """
        Note:
          The modified attributes and according operations are shown below:
            ==================================  ===========================
            Attribute                           Operation
            ==================================  ===========================
            ``ctx.data_batch``                  Initialize batch data
            ==================================  ===========================
        """
        # prepare data batch
        try:
            batch_size = ctx.cfg.dataloader.batch_size
            is_final_batch = (ctx.cur_batch_i == (ctx.batch_num - 1))

            if not is_final_batch:
                idx = ctx.all_idx[
                      ctx.cur_batch_i * batch_size: (ctx.cur_batch_i + 1) * batch_size]
            else:
                idx = ctx.all_idx[ctx.cur_batch_i * batch_size:]

            ctx.idx = CtxVar(idx, LIFECYCLE.BATCH)
            ctx.batch_size = CtxVar(len(idx), LIFECYCLE.BATCH)

            # ctx.data_batch = CtxVar(
            #     next(ctx.get("{}_loader".format(ctx.cur_split))),
            #     LIFECYCLE.BATCH)

        except StopIteration:
            raise StopIteration

    def _hook_on_batch_forward(self, ctx):
        """
        Note:
          The modified attributes and according operations are shown below:
            ==================================  ===========================
            Attribute                           Operation
            ==================================  ===========================
            ``ctx.y_true``                      Move to `ctx.device`
            ``ctx.y_prob``                      Forward propagation get y_prob
            ``ctx.loss_batch``                  Calculate the loss
            ``ctx.batch_size``                  Get the batch_size
            ==================================  ===========================
        """
        # x, label = [_.to(ctx.device) for _ in ctx.data_batch]


        output = ctx.model(ctx.x, ctx.adj, ctx.idx, ctx.subgraphs, ctx.batch_size)
        ctx.output = CtxVar(output, LIFECYCLE.BATCH)

        loss = loss_function(output, ctx.batch_size,ctx.cfg.model.negsamp_ratio)
        loss = torch.mean(loss)
        ctx.loss_batch = CtxVar(loss, LIFECYCLE.BATCH)
        # ctx.epoch_loss += loss.item() * ctx.batch_size



        # pred = ctx.model(x)
        # if len(label.size()) == 0:
        #     label = label.unsqueeze(0)
        #
        # ctx.y_true = CtxVar(label, LIFECYCLE.BATCH)
        # ctx.y_prob = CtxVar(pred, LIFECYCLE.BATCH)
        # ctx.loss_batch = CtxVar(ctx.criterion(pred, label), LIFECYCLE.BATCH)
        # ctx.batch_size = CtxVar(len(label), LIFECYCLE.BATCH)
    def _hook_on_batch_end(self, ctx):
        """
        Note:
          The modified attributes and according operations are shown below:
            ==================================  ===========================
            Attribute                           Operation
            ==================================  ===========================
            ``ctx.num_samples``                 Add ``ctx.batch_size``
            ``ctx.loss_batch_total``            Add batch loss
            ``ctx.loss_regular_total``          Add batch regular loss
            ``ctx.ys_true``                     Append ``ctx.y_true``
            ``ctx.ys_prob``                     Append ``ctx.ys_prob``
            ==================================  ===========================
        """
        # compute the anomaly score
        logits = torch.sigmoid(torch.squeeze(ctx.output))
        ano_score = - (logits[:ctx.batch_size] - logits[ctx.batch_size:]).detach().cpu().numpy()
        ctx.decision_scores[ctx.idx] = ano_score

        # update statistics
        ctx.num_samples_batch += ctx.batch_size
        ctx.loss_batch_total += ctx.loss_batch.item() * ctx.batch_size
        ctx.loss_regular_total += float(ctx.get("loss_regular", 0.))
        # cache label for evaluate
        # ctx.ys_true.append(ctx.y_true.detach().cpu().numpy())
        # ctx.ys_prob.append(ctx.y_prob.detach().cpu().numpy())

    def _hook_on_epoch_end(self, ctx):
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
        # ctx.ys_true = CtxVar(np.concatenate(ctx.ys_true), LIFECYCLE.ROUTINE)
        # ctx.ys_prob = CtxVar(np.concatenate(ctx.ys_prob), LIFECYCLE.ROUTINE)
        # results = ctx.monitor.eval(ctx)
        # setattr(ctx, 'eval_metrics', results)
        ctx.loss_epoch_total += ctx.loss_batch_total
        ctx.num_samples += ctx.num_samples_batch
        epoch = self.ctx.cur_epoch_i
        ctx.multi_epoch_ano_score[epoch, :] = ctx.decision_scores
        # auc=None
        if ctx.cfg.model.verbose:
            print("Epoch {}: Loss {:.6f}".format(epoch,ctx.loss_batch_total /ctx.num_samples_batch), end='')
            if ctx.y_true is not None:
                # remove the mend part
                mend_idx = torch.where(ctx.y_true == 3)[0]
                mask = ctx.y_true != 3
                ctx.y_true = torch.masked_select(ctx.y_true, mask)
                ctx.decision_scores = np.delete(ctx.decision_scores, mend_idx)
                # prediction = (ctx.decision_scores > ctx.threshold_).astype(int).ravel()
                # compute auc
                roc_auc,fpr, tpr = result_auc(ctx.y_true.numpy(), ctx.decision_scores)
                # compute acc
                print(" | roc_auc {:.6f}".format( roc_auc), end='')
                # print(" | ACC {:.6f}".format(acc), end='')
                # print(" | Recall {:.6f}".format(recall), end='')
            print()


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

        ano_score_final = np.mean(ctx.multi_epoch_ano_score, axis=0)
        ctx.decision_scores_ = CtxVar(ano_score_final, LIFECYCLE.ROUTINE)
        threshold_ = self._process_decision_scores(ctx, ano_score_final)  # setting threshold_
        # remove the mend part
        mend_idx = torch.where(ctx.data['data'].ay == 3)[0]
        mask = ctx.data['data'].ay != 3
        y_true = torch.masked_select(ctx.data['data'].ay, mask)
        ano_score_final = np.delete( ano_score_final, mend_idx)
        prediction = (ano_score_final > threshold_).astype(int).ravel()
        roc_auc,fpr, tpr = result_auc(y_true, ano_score_final)
        acc,recall = result_acc_and_rec(y_true, prediction)
        # compute acc
        print("AUC {:.6f}".format(roc_auc), end='')
        print(" | ACC {:.6f}".format(acc), end='')
        print(" | Recall {:.6f}".format(recall), end='')
        print()
        # auc_score = eval_roc_auc(ctx.data['data'].ay, ano_score_final)
        # print('AUC Score:', auc_score)

        # G = ctx.data['data']
        # pred_score = self.decision_function(ctx, G)
        # # prediction = (pred_score > ctx.threshold_).astype(int).ravel()
        # auc_score = eval_roc_auc(G.ay.numpy(), pred_score)
        # print('Predict------AUC Score:', auc_score)

        # just store the value of the last round

        results = {
            f"{ctx.cur_mode}_avg_loss": ctx.loss_epoch_total / ctx.num_samples,
            f"{ctx.cur_mode}_roc_auc": roc_auc,
            f"{ctx.cur_mode}_acc": acc,
            f"{ctx.cur_mode}_recall": recall,
            f"{ctx.cur_mode}_total": ctx.num_samples,
            f"{ctx.cur_mode}_fpr": fpr,
            f"{ctx.cur_mode}_tpr": tpr
        }

        setattr(ctx, 'eval_metrics', results)



        # return prediction, confidence

    def _hook_on_predict(self, ctx):
        confidence = None
        ctx.model.to(ctx.device)
        G = ctx.data['data']
        num_samples = len(G.ay)
        ctx.num_samples = CtxVar(num_samples, LIFECYCLE.ROUTINE)
        # check_is_fitted(ctx, ['decision_scores_', 'threshold_', 'labels_'])
        pred_score = self.decision_function(ctx,G)

        # remove the mend part
        mend_idx = torch.where(G.ay == 3)[0]
        mask = G.ay != 3
        y_true = torch.masked_select(G.ay, mask)
        pred_score = np.delete(pred_score, mend_idx)
        threshold_ = self._process_decision_scores(ctx, pred_score )
        prediction = (pred_score > threshold_).astype(int).ravel()
        roc_auc,fpr, tpr = result_auc(y_true,  pred_score)
        acc, recall = result_acc_and_rec(y_true, prediction)

        # compute acc
        print("AUC {:.6f}".format(roc_auc), end='')
        print(" | ACC {:.6f}".format(acc), end='')
        print(" | Recall {:.6f}".format(recall), end='')
        print()
        # auc_score = eval_roc_auc(ctx.data['data'].ay, ano_score_final)
        results = {
            # f"{ctx.cur_mode}_avg_loss": ctx.loss_epoch_total / num_samples,
            f"{ctx.cur_mode}_roc_auc": roc_auc,
            f"{ctx.cur_mode}_acc": acc,
            f"{ctx.cur_mode}_recall": recall,
            f"{ctx.cur_mode}_total":ctx.num_samples,
            f"{ctx.cur_mode}_fpr": fpr,
            f"{ctx.cur_mode}_tpr": tpr
        }

        setattr(ctx, 'eval_metrics', results)

    # def _hook_on_decision_function(self,ctx,rounds = 10):
    #     x, adj = self.process_graph(G)
    #
    #     if self.batch_size:
    #         batch_num = self.num_nodes // self.batch_size + 1
    #     else:  # full batch training
    #         batch_num = 1
    #
    #     multi_round_ano_score = np.zeros((rounds, self.num_nodes))
    #
    #     # enable the evaluation mode
    #     self.model.eval()
    #
    #     for r in range(rounds):
    #
    #         all_idx = list(range(self.num_nodes))
    #         random.shuffle(all_idx)
    #
    #         subgraphs = generate_rw_subgraph(G, self.num_nodes,
    #                                          self.subgraph_size)
    #
    #         for batch_idx in range(batch_num):
    #
    #             is_final_batch = (batch_idx == (batch_num - 1))
    #
    #             if not is_final_batch:
    #                 idx = all_idx[
    #                       batch_idx *
    #                       self.batch_size: (batch_idx + 1) * self.batch_size]
    #             else:
    #                 idx = all_idx[batch_idx * self.batch_size:]
    #
    #             cur_batch_size = len(idx)
    #
    #             with torch.no_grad():
    #                 output = self.model(x, adj, idx, subgraphs, cur_batch_size)
    #             logits = torch.sigmoid(torch.squeeze(output))
    #             ano_score = - (logits[:cur_batch_size] - logits[
    #                                                      cur_batch_size:]).cpu().numpy()
    #             multi_round_ano_score[r, idx] = ano_score
    #
    #     ano_score_final = np.mean(multi_round_ano_score, axis=0)
    #
    #     return ano_score_final

    def _process_decision_scores(self,ctx,decision_scores_):
        """Internal function to calculate key attributes:
        - threshold_: used to decide the binary label
        - labels_: binary labels of training data
        Returns
        -------
        self
        """

        threshold_ = percentile(decision_scores_,
                                     100 * (1 - ctx.cfg.model.contamination))
        # ctx.threshold_ = CtxVar(threshold_, LIFECYCLE.ROUTINE)

        labels_ = (decision_scores_ > threshold_).astype(
            'int').ravel()
        # ctx.labels_ = CtxVar(labels_, LIFECYCLE.ROUTINE)


        # calculate for predict_proba()

        # ctx._mu = CtxVar(np.mean(decision_scores_), LIFECYCLE.ROUTINE)
        # ctx._sigma = CtxVar(np.std(decision_scores_), LIFECYCLE.ROUTINE)

        return  threshold_

    def decision_function(self, ctx, G, rounds=100):
        num_nodes, feat_dim, x, adj = process_graph(G, ctx.device)

        if ctx.cfg.dataloader.batch_size:
            batch_num = math.ceil(num_nodes / ctx.cfg.dataloader.batch_size)
        else:  # full batch training
            batch_num = 1

        multi_round_ano_score = np.zeros((rounds, num_nodes))

        # enable the evaluation mode
        ctx.model.eval()
        batch_size = ctx.cfg.dataloader.batch_size
        for r in range(rounds):

            all_idx = list(range(num_nodes))
            random.shuffle(all_idx)

            subgraphs = generate_rw_subgraph(G, num_nodes, ctx.cfg.model.subgraph_size)

            for batch_idx in range(batch_num):

                is_final_batch = (batch_idx == (batch_num - 1))

                if not is_final_batch:
                    idx = all_idx[
                          batch_idx *
                          batch_size: (batch_idx + 1) * batch_size]
                else:
                    idx = all_idx[batch_idx * batch_size:]

                cur_batch_size = len(idx)

                with torch.no_grad():
                    output = ctx.model(x, adj, idx, subgraphs, cur_batch_size)
                logits = torch.sigmoid(torch.squeeze(output))
                ano_score = - (logits[:cur_batch_size] - logits[
                                                         cur_batch_size:]).cpu().numpy()
                multi_round_ano_score[r, idx] = ano_score

        ano_score_final = np.mean(multi_round_ano_score, axis=0)

        return ano_score_final

    def predict_confidence(self, ctx, G):
        """Predict the model's confidence in making the same prediction
        under slightly different training sets.
        See :cite:`perini2020quantifying`.

        Parameters
        ----------
        G : PyTorch Geometric Data instance (torch_geometric.data.Data)
            The input graph.

        Returns
        -------
        confidence : numpy array of shape (n_samples,)
            For each observation, tells how consistently the model would
            make the same prediction if the training set was perturbed.
            Return a probability, ranging in [0,1].

        """

        check_is_fitted(ctx, ['decision_scores_', 'threshold_', 'labels_'])

        n = len(ctx.decision_scores_)

        # todo: this has an optimization opportunity since the scores may
        # already be available
        test_scores = self.decision_function(G)

        count_instances = np.vectorize(
            lambda x: np.count_nonzero(ctx.decision_scores_ <= x))
        n_instances = count_instances(test_scores)

        # Derive the outlier probability using Bayesian approach
        posterior_prob = np.vectorize(lambda x: (1 + x) / (2 + n))(n_instances)

        # Transform the outlier probability into a confidence value
        confidence = np.vectorize(
            lambda p: 1 - binom.cdf(n - int(n * ctx.cfg.model.contamination), n, p))(
            posterior_prob)
        prediction = (test_scores > ctx.threshold_).astype('int').ravel()
        np.place(confidence, prediction == 0, 1 - confidence[prediction == 0])

        return confidence
