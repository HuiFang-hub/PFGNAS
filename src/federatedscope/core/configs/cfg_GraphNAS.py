from src.federatedscope.core.configs.config import CN
from src.federatedscope.register import register_config
import time

def extend_model_cfg(cfg):
    # ---------------------------------------------------------------------- #
    # graphnas related options
    # ---------------------------------------------------------------------- #
    cfg.graphnas = CN()
    cfg.graphnas.save_epoch = 2
    cfg.graphnas.max_save_num = 5

    cfg.graphnas.controller = CN()
    cfg.graphnas.controller.layers_of_child_model = 2
    cfg.graphnas.controller.shared_initial_step = 0
    cfg.graphnas.controller.entropy_mode = 'reward'
    cfg.graphnas.controller.entropy_coeff = 1e-4
    cfg.graphnas.controller.shared_rnn_max_length = 35
    cfg.graphnas.controller.search_mode = 'macro'
    cfg.graphnas.controller.format = 'two'
    cfg.graphnas.controller.max_epoch = 10
    cfg.graphnas.controller.ema_baseline_decay = 0.95
    cfg.graphnas.controller.discount =1.0
    cfg.graphnas.controller.controller_max_step =100
    cfg.graphnas.controller.controller_optim = 'adam'
    cfg.graphnas.controller.controller_lr = 3.5e-4
    cfg.graphnas.controller.controller_grad_clip= 0.0
    cfg.graphnas.controller.tanh_c = 2.5
    cfg.graphnas.controller.softmax_temperature = 5.0
    cfg.graphnas.controller.derive_num_sample = 100
    cfg.graphnas.controller.derive_finally = True
    cfg.graphnas.controller.derive_from_history = True

    cfg.graphnas.child = CN()
    cfg.graphnas.child.retrain_epochs = 300
    cfg.graphnas.child.residual = 'store_false'
    cfg.graphnas.child.indrop = 0.6
    cfg.graphnas.child.lr = 0.005
    cfg.graphnas.child.param_file = "cora_test.pkl"
    cfg.graphnas.child.optim_file = "opt_cora_test.pkl"
    cfg.graphnas.child.max_param = 5e6
    cfg.graphnas.child.supervised = False
    cfg.graphnas.child.submanager_log_file = f"sub_manager_logger_file_{time.time()}.txt"

    # --------------- register corresponding check function ----------
    cfg.register_cfg_check_fun(assert_model_cfg)


def assert_model_cfg(cfg):
    pass


register_config("graphnas", extend_model_cfg)
