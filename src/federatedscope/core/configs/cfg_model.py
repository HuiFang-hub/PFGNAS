from src.federatedscope.core.configs.config import CN
from src.federatedscope.register import register_config


def extend_model_cfg(cfg):
    # ---------------------------------------------------------------------- #
    # Model related options
    # ---------------------------------------------------------------------- #
    cfg.model = CN()

    cfg.model.model_num_per_trainer = 1  # some methods may leverage more
    # than one model in each trainer
    cfg.model.type = 'lr'
    cfg.model.use_bias = True
    cfg.model.task = 'node'
    cfg.model.hidden = 256
    cfg.model.dropout = 0.5
    cfg.model.in_channels = 0  # If 0, model will be built by data.shape
    cfg.model.num_classes = 1
    cfg.model.layer = 2  # In GPR-GNN, K = layer
    cfg.model.graph_pooling = 'mean'
    cfg.model.embed_size = 8
    cfg.model.num_item = 0
    cfg.model.num_user = 0
    cfg.model.input_shape = ()  # A tuple, e.g., (in_channel, h, w)

    # For tree-based model
    cfg.model.lambda_ = 0.1
    cfg.model.gamma = 0
    cfg.model.num_of_trees = 10
    cfg.model.max_tree_depth = 3

    # language model for hetero NLP tasks
    cfg.model.stage = ''  # ['assign', 'contrast']
    cfg.model.model_type = 'google/bert_uncased_L-2_H-128_A-2'
    cfg.model.pretrain_tasks = []
    cfg.model.downstream_tasks = []
    cfg.model.num_labels = 1
    cfg.model.max_length = 200
    cfg.model.min_length = 1
    cfg.model.no_repeat_ngram_size = 3
    cfg.model.length_penalty = 2.0
    cfg.model.num_beams = 5
    cfg.model.label_smoothing = 0.1
    cfg.model.n_best_size = 20
    cfg.model.max_answer_len = 30
    cfg.model.null_score_diff_threshold = 0.0
    cfg.model.use_contrastive_loss = False
    cfg.model.contrast_topk = 100
    cfg.model.contrast_temp = 1.0

    # ---------------------------------------------------------------------- #
    # NAS
    # ---------------------------------------------------------------------- #
    cfg.model.struct = ' '
    cfg.model.operations = 'abcd'
    cfg.model.actions = 'a'


    # ---------------------------------------------------------------------- #
    # Criterion related options
    # ---------------------------------------------------------------------- #
    cfg.criterion = CN()
    cfg.criterion.type = 'MSELoss'
    cfg.criterion.model_type = 'CrossEntropyLoss'
    cfg.criterion.federated_type = 'BCEWithLogitsLoss'
    # ---------------------------------------------------------------------- #
    # regularizer related options
    # ---------------------------------------------------------------------- #
    cfg.regularizer = CN()
    cfg.regularizer.type = ''
    cfg.regularizer.mu = 0.

    # -----------------------------------------------------------------------#
    # custom define (cola)
    # -----------------------------------------------------------------------#
    cfg.model.subgraph_size = 4
    cfg.model.negsamp_ratio = 1
    cfg.model.verbose = False
    cfg.model.contamination = 0.1
    cfg.model.return_confidenc = False
    cfg.model.lr = 1e-3
    cfg.model.weight_decay=0.
    # -----------------------------------------------------------------------#
    # custom define (anemone)
    # -----------------------------------------------------------------------#
    cfg.model.negsamp_ratio_patch = 1
    cfg.model.negsamp_ratio_context = 1
    cfg.model.alpha = 1.0
    cfg.model.test_rounds = 100



    # --------------- register corresponding check function ----------
    cfg.register_cfg_check_fun(assert_model_cfg)


def assert_model_cfg(cfg):
    pass


register_config("model", extend_model_cfg)
