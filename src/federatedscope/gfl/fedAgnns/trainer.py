import torch
from torch import optim
from src.federatedscope.core.trainers import GeneralTorchTrainer
from src.federatedscope.core.trainers.context import CtxVar
from src.federatedscope.core.trainers.enums import MODE, LIFECYCLE
from src.federatedscope.core.monitors import Monitor
from src.federatedscope.register import register_trainer
from src.autogl.module.nas.utils import replace_layer_choice, replace_input_choice
from src.federatedscope.gfl.fedDarts.utils import DartsLayerChoice, DartsInputChoice
import collections
import copy
import logging

from src.federatedscope.core.trainers.base_trainer import BaseTrainer
from src.federatedscope.core.trainers.enums import MODE, LIFECYCLE
from src.federatedscope.core.auxiliaries.decorators import use_diff
from src.federatedscope.core.trainers.utils import format_log_hooks, \
    filter_by_specified_keywords
from src.federatedscope.core.trainers.context import Context, CtxVar, lifecycle
import logging
logger = logging.getLogger(__name__)

class fedAggns_trainer(GeneralTorchTrainer):

    HOOK_TRIGGER = [
        "on_fit_start", "on_epoch_start", "on_batch_start", "on_batch_forward",
        "on_batch_backward", "on_batch_end", "on_epoch_end", "on_fit_end"
    ]

    def __init__(self,
                 model,
                 data,
                 device,
                 config,
                 only_for_eval=False,
                 monitor=None):
        self._cfg = config
        # model = model.to(device)
        self.device = device
        self.ctx = Context(model, self.cfg, data, device)

        # Parse data and setup init vars in ctx
        self._setup_data_related_var_in_ctx(self.ctx)

        assert monitor is not None, \
            f"Monitor not found in trainer with class {type(self)}"
        self.ctx.monitor = monitor
        # the "model_nums", and "models" are used for multi-model case and
        # model size calculation
        self.model_nums = 1
        self.ctx.models = [model]
        # "mirrored_models": whether the internal multi-models adopt the
        # same architects and almost the same behaviors,
        # which is used to simply the flops, model size calculation
        self.ctx.mirrored_models = False

        # Atomic operation during training/evaluation
        self.hooks_in_train = collections.defaultdict(list)

        # By default, use the same trigger keys
        self.hooks_in_eval = copy.deepcopy(self.hooks_in_train)
        self.hooks_in_ft = copy.deepcopy(self.hooks_in_train)

        # register necessary hooks into self.hooks_in_train and
        # self.hooks_in_eval
        if not only_for_eval:
            self.register_default_hooks_train()
        if self.cfg.finetune.before_eval:
            self.register_default_hooks_ft()
        self.register_default_hooks_eval()

        if self.cfg.federate.mode == 'distributed':
            self.print_trainer_meta_info()
        else:
            # in standalone mode, by default, we print the trainer info only
            # once for better logs readability
            pass
        self.ctx.nas_modules = []
        self.ctx.model_optim = torch.optim.Adam(model.parameters(), lr=self.cfg.train.optimizer.lr, weight_decay=5e-4)

        replace_layer_choice(model, DartsLayerChoice,self.ctx.nas_modules)
        replace_input_choice(model, DartsInputChoice, self.ctx.nas_modules)
        ctrl_params = {}
        for _, m in self.ctx.nas_modules:
            if m.name in ctrl_params:
                assert (
                        m.alpha.size() == ctrl_params[m.name].size()
                ), "Size of parameters with the same label should be same."
                m.alpha = ctrl_params[m.name]
            else:
                ctrl_params[m.name] = m.alpha
        para = list(ctrl_params.values())
        self.ctx.arch_optim = torch.optim.Adam(para, lr=self.ctx.cfg.train.optimizer.lr, weight_decay=5e-4)
        # test = 1
        # ctx.model_optim = CtxVar(model_optim , LIFECYCLE.ROUTINE)
        # self.ctx.arch_optim = CtxVar(arch_optim, LIFECYCLE.ROUTINE)


    def parse_data(self, data):
        """Populate "{}_data", "{}_loader" and "num_{}_data" for different
        modes
        """
        init_dict = dict()
        if isinstance(data, dict):
            for mode in ["train", "val", "test"]:
                init_dict["{}_loader".format(mode)] = data.get(mode)

                # da =data.get(mode)
                # train_label = da.y[da['train_mask']]
                # val_label = da.y[da['val_mask']]
                # test_label = da.y[da['test_mask']]
                # print(f'train_label :{train_label}\n val_label:{val_label}\n test_label:{test_label}')

                init_dict["{}_data".format(mode)] = None
                # For node-level task dataloader contains one graph
                init_dict["num_{}_data".format(mode)] = 1  # self.cfg.dataloader.batch_size
        else:
            raise TypeError("Type of data should be dict.")
        return init_dict

    def get_last_models(self,ctx):
        selection = self.export(ctx.nas_modules)
        return ctx.model.parse_model(selection, self.device)


    @torch.no_grad()
    def export(self, nas_modules) -> dict:
        result = dict()
        for name, module in nas_modules:
            if name not in result:
                result[name] = module.export()
        return result

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
        # ctx.model.to(ctx.device)
        # ctx.model.train()

        # if ctx.cur_mode == 'train':

            # ctx.optimizer = get_optimizer(ctx.model,
            #                               **ctx.cfg[ctx.cur_mode].optimizer)
        ctx.loss_batch_total = CtxVar(0., LIFECYCLE.ROUTINE)
        ctx.loss_regular_total = CtxVar(0., LIFECYCLE.ROUTINE)
        ctx.num_samples = CtxVar(0, LIFECYCLE.ROUTINE)
        ctx.ys_true = CtxVar([], LIFECYCLE.ROUTINE)
        ctx.ys_prob = CtxVar([], LIFECYCLE.ROUTINE)

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

            ctx.data_batch = CtxVar(next(ctx.get("{}_loader".format(ctx.cur_split))), LIFECYCLE.BATCH)
            # da = ctx.data_batch
            # train_label = da.y[da['train_mask']]
            # val_label = da.y[da['val_mask']]
            # test_label = da.y[da['test_mask']]
            # print(f'train_label :{train_label}\n val_label:{val_label}\n test_label:{test_label}')

            # text = ctx.data_batch
            # a =1
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
        batch = ctx.data_batch.to(ctx.device)
        # if ctx.cur_split=='train' or ctx.cur_split=='val':
        sm_idx = 0  # ;st_time=time.time()
        y_predict = None
        for supermask in ctx.cfg.data.supermasks:
            ctx.model = ctx.model.to(ctx.device)
            y_predict = ctx.model(batch, supermask)[batch[f'{ctx.cur_split}_mask']]
            if sm_idx == 0:
                exec('loss_{}=F.cross_entropy(y_predict,batch.y[batch[{}_mask]])'.format(sm_idx,ctx.cur_split))
            else:
                exec(
                    'loss_{}=loss_{} + F.cross_entropy(y_predict,batch.y[batch[{}_mask]])'.format(sm_idx, sm_idx - 1,ctx.cur_split))
            sm_idx += 1
        factor = len(ctx.cfg.data.supermasks)
        ctx.loss_batch = CtxVar(eval('loss_{}'.format(sm_idx - 1)) / factor, LIFECYCLE.BATCH)
        ctx.y_true = CtxVar(y_predict, LIFECYCLE.BATCH)
        pred = ctx.model(batch)[batch[f'{ctx.cur_split}_mask']]
        ctx.y_prob = CtxVar(pred, LIFECYCLE.BATCH)

        # else:
            # selection = self.export(ctx.nas_modules)
            # ctx.model = ctx.model.parse_model(selection, self.device)._model.to(ctx.device)  # note this
            # pred = ctx.model(batch)[batch[f'{ctx.cur_split}_mask']]
            # label = batch.y[batch[f'{ctx.cur_split}_mask']]
            # ctx.batch_size = torch.sum(ctx.data_batch['{}_mask'.format(
            #     ctx.cur_split)]).item()
            # ctx.loss_batch = CtxVar(ctx.criterion(pred, label), LIFECYCLE.BATCH)
            # ctx.y_true = CtxVar(label, LIFECYCLE.BATCH)
            # ctx.y_prob = CtxVar(pred, LIFECYCLE.BATCH)

            # ctx.model = ctx.model.to(ctx.device)



    def _hook_on_batch_backward(self, ctx):
        """
        Note:
          The modified attributes and according operations are shown below:
            ==================================  ===========================
            Attribute                           Operation
            ==================================  ===========================
            ``ctx.optimizer``                   Update by gradient
            ``ctx.loss_task``                   Backward propagation
            ``ctx.scheduler``                   Update by gradient
            ==================================  ===========================
        """
        ctx.arch_optim.zero_grad()
        ctx.arch_loss_batch.backward()
        # ctx.arch_optim.step()

        ctx.model_optim.zero_grad()
        ctx.loss_batch.backward()
        if ctx.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(ctx.model.parameters(),
                                           ctx.grad_clip)
        # ctx.model_optim.step()

        # if ctx.scheduler is not None:
        #     ctx.scheduler.step()

    def _hook_on_batch_forward_flop_count(self, ctx):
        if not isinstance(self.ctx.monitor, Monitor):
            logger.warning(
                f"The trainer {type(self)} does contain a valid monitor, "
                f"this may be caused by "
                f"initializing trainer subclasses without passing a valid "
                f"monitor instance."
                f"Plz check whether this is you want.")
            return

        if self.cfg.eval.count_flops and self.ctx.monitor.flops_per_sample \
                == 0:
            # calculate the flops_per_sample
            try:
                batch = ctx.data_batch.to(ctx.device)
                from torch_geometric.data import Data
                if isinstance(batch, Data):
                    x, edge_index = batch.x, batch.edge_index
                from fvcore.nn import FlopCountAnalysis
                flops_one_batch = FlopCountAnalysis(ctx.model,
                                                    (x, edge_index)).total()

                if self.model_nums > 1 and ctx.mirrored_models:
                    flops_one_batch *= self.model_nums
                    logger.warning(
                        "the flops_per_batch is multiplied by "
                        "internal model nums as self.mirrored_models=True."
                        "if this is not the case you want, "
                        "please customize the count hook")
                self.ctx.monitor.track_avg_flops(flops_one_batch,
                                                 ctx.batch_size)
            except:
                logger.warning(
                    "current flop count implementation is for general "
                    "NodeFullBatchTrainer case: "
                    "1) the ctx.model takes only batch = ctx.data_batch as "
                    "input."
                    "Please check the forward format or implement your own "
                    "flop_count function")
                self.ctx.monitor.flops_per_sample = -1  # warning at the
                # first failure

        # by default, we assume the data has the same input shape,
        # thus simply multiply the flops to avoid redundant forward
        self.ctx.monitor.total_flops += self.ctx.monitor.flops_per_sample * \
            ctx.batch_size

    def train(self, target_data_split_name="train", hooks_set=None):
        hooks_set = hooks_set or self.hooks_in_train

        self.ctx.check_split(target_data_split_name)

        num_samples = self._run_routine(MODE.TRAIN, hooks_set,
                                        target_data_split_name)
        return num_samples, self.get_model_para(), self.ctx.eval_metrics, self.get_last_models(self.ctx)._model

def  call_fedAggns_trainer(trainer_type):
    if trainer_type == 'fedAggns_trainer':
        trainer_builder = fedAggns_trainer
    else:
        trainer_builder = None

    return trainer_builder

register_trainer('fedDarts_trainer', call_fedAggns_trainer)