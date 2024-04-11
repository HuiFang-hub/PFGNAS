import glob
import os

import numpy as np
import scipy.signal
import torch

from src.graphnas_variants.macro_graphnas.pyg.pyg_gnn_model_manager import GeoCitationManager
import src.graphnas.utils.tensor_utils as utl
from src.graphnas.gnn_model_manager import CitationGNNManager


logger = utl.get_logger()


def discount(x, amount):
    return scipy.signal.lfilter([1], [1, -amount], x[::-1], axis=0)[::-1]


history = []


def scale(value, last_k=10, scale_value=1):
    '''
    scale value into [-scale_value, scale_value], according last_k history
    '''
    max_reward = np.max(history[-last_k:])
    if max_reward == 0:
        return value
    return scale_value / max_reward * value


def _get_optimizer(name):
    if name.lower() == 'sgd':
        optim = torch.optim.SGD
    elif name.lower() == 'adam':
        optim = torch.optim.Adam

    return optim


class Trainer(object):
    """Manage the training process"""

    def __init__(self, cfg,data):
        """
        Constructor for training algorithm.
        Build sub-model manager and controller.
        Build optimizer and cross entropy loss for controller.

        cfg:
            cfg: From command line, picked up by `argparse`.
        """
        self.cfg = cfg
        self.controller_step = 0  # counter for controller
        self.device = cfg.device
        self.epoch = 0
        self.start_epoch = 0
        self.data = data
        self.max_length = self.cfg.graphnas.controller.shared_rnn_max_length

        self.with_retrain = False
        self.submodel_manager = None
        self.controller = None
        self.build_model()  # build controller and sub-model

        controller_optimizer = _get_optimizer(self.cfg.graphnas.controller.controller_optim)
        self.controller_optim = controller_optimizer(self.controller.parameters(), lr=self.cfg.train.optimizer.lr)

        # if self.cfg.mode == "derive":
        #     self.load_model()

    def build_model(self):
        self.share_param = False
        self.with_retrain = True
        self.shared_initial_step = 0
        if self.cfg.graphnas.controller.search_mode == "macro":
            # generate model description in macro way (generate entire network description)
            from src.graphnas.search_space import MacroSearchSpace
            search_space_cls = MacroSearchSpace()
            self.search_space = search_space_cls.get_search_space()
            self.action_list = search_space_cls.generate_action_list(self.cfg.graphnas.controller.layers_of_child_model)
            # build RNN controller
            from src.graphnas.graphnas_controller import SimpleNASController
            self.controller = SimpleNASController(self.cfg.graphnas, action_list=self.action_list,
                                                  search_space=self.search_space,device = self.cfg.device
                                                  ).to(self.cfg.device)

            # if self.cfg.dataset in ["cora", "citeseer", "pubmed"]:
            #     # implements based on dgl
            #     self.submodel_manager = CitationGNNManager(self.cfg)
            if self.cfg.data.type in ["cora", "citeseer", "pubmed"]:
                # implements based on pyg
                self.submodel_manager = GeoCitationManager(self.data,self.cfg)


        # if self.cfg.search_mode == "micro":
        #     self.cfg.graphnas.controller.format = "micro"
        #     self.cfg.predict_hyper = True
        #     if not hasattr(self.cfg, "num_of_cell"):
        #         self.cfg.num_of_cell = 2
        #     from graphnas_variants.micro_graphnas.micro_search_space import IncrementSearchSpace
        #     search_space_cls = IncrementSearchSpace()
        #     search_space = search_space_cls.get_search_space()
        #     from graphnas.graphnas_controller import SimpleNASController
        #     from graphnas_variants.micro_graphnas.micro_model_manager import MicroCitationManager
        #     self.submodel_manager = MicroCitationManager(self.cfg)
        #     self.search_space = search_space
        #     action_list = search_space_cls.generate_action_list(cell=self.cfg.num_of_cell)
        #     if hasattr(self.cfg, "predict_hyper") and self.cfg.predict_hyper:
        #         self.action_list = action_list + ["learning_rate", "dropout", "weight_decay", "hidden_unit"]
        #     else:
        #         self.action_list = action_list
        #     self.controller = SimpleNASController(self.cfg, action_list=self.action_list,
        #                                           search_space=self.search_space,
        #                                           ).to(self.cfg.device)
        #     if self.cuda:
        #         self.controller.cuda()
        #
        # if self.cuda:
        #     self.controller.cuda()

    def form_gnn_info(self, gnn):
        if self.cfg.search_mode == "micro":
            actual_action = {}
            if self.cfg.predict_hyper:
                actual_action["action"] = gnn[:-4]
                actual_action["hyper_param"] = gnn[-4:]
            else:
                actual_action["action"] = gnn
                actual_action["hyper_param"] = [0.005, 0.8, 5e-5, 128]
            return actual_action
        return gnn

    def train(self):
        """
        Each epoch consists of two phase:
        - In the first phase, shared parameters are trained to exploration.
        - In the second phase, the controller's parameters are trained.
        """

        for self.epoch in range(self.start_epoch, self.cfg.graphnas.controller.max_epoch):
            # 1. Training the shared parameters of the child graphnas
            self.train_shared(max_step=self.shared_initial_step)
            # 2. Training the controller parameters theta
            self.train_controller()
            # 3. Derive architectures
            self.derive(sample_num=self.cfg.derive_num_sample)

            if self.epoch % self.cfg.save_epoch == 0:
                self.save_model()

        if self.cfg.derive_finally:
            best_actions = self.derive()
            print("best structure:" + str(best_actions))
        self.save_model()

    def train_shared(self, max_step=50, gnn_list=None):
        """
        cfg:
            max_step: Used to run extra training steps as a warm-up.
            gnn: If not None, is used instead of calling sample().

        """
        if max_step == 0:  # no train shared
            return

        print("*" * 35, "training model", "*" * 35)
        gnn_list = gnn_list if gnn_list else self.controller.sample(max_step)

        for gnn in gnn_list:
            gnn = self.form_gnn_info(gnn)
            try:
                _, val_score = self.submodel_manager.train(gnn, format=self.cfg.graphnas.controller.format)
                logger.info(f"{gnn}, val_score:{val_score}")
            except RuntimeError as e:
                if 'CUDA' in str(e):  # usually CUDA Out of Memory
                    print(e)
                else:
                    raise e

        print("*" * 35, "training over", "*" * 35)

    def get_reward(self, gnn_list, entropies, hidden):
        """
        Computes the reward of a single sampled model on validation data.
        """
        if not isinstance(entropies, np.ndarray):
            entropies = entropies.data.cpu().numpy()
        if isinstance(gnn_list, dict):
            gnn_list = [gnn_list]
        if isinstance(gnn_list[0], list) or isinstance(gnn_list[0], dict):
            pass
        else:
            gnn_list = [gnn_list]  # when structure_list is one structure

        reward_list = []
        for gnn in gnn_list:
            gnn = self.form_gnn_info(gnn)
            reward = self.submodel_manager.test_with_param(gnn, format=self.cfg.graphnas.controller.format,
                                                           with_retrain=self.with_retrain)  ####

            if reward is None:  # cuda error hanppened
                reward = 0
            else:
                reward = reward[1]

            reward_list.append(reward)

        if self.cfg.entropy_mode == 'reward':
            rewards = reward_list + self.cfg.entropy_coeff * entropies
        elif self.cfg.entropy_mode == 'regularizer':
            rewards = reward_list * np.ones_like(entropies)
        else:
            raise NotImplementedError(f'Unkown entropy mode: {self.cfg.entropy_mode}')

        return rewards, hidden

    def train_controller(self):
        """
            Train controller to find better structure.
        """
        print("*" * 35, "training controller", "*" * 35)
        model = self.controller
        model.train()

        baseline = None
        adv_history = []
        entropy_history = []
        reward_history = []

        hidden = self.controller.init_hidden(self.cfg.dataloader.batch_size)
        total_loss = 0
        for step in range(self.cfg.graphnas.controller.controller_max_step):
            # sample graphnas
            structure_list, log_probs, entropies = self.controller.sample(with_details=True)

            # calculate reward
            np_entropies = entropies.data.cpu().numpy()
            results = self.get_reward(structure_list, np_entropies, hidden)
            torch.cuda.empty_cache()

            if results:  # has reward
                rewards, hidden = results
            else:
                continue  # CUDA Error happens, drop structure and step into next iteration

            # discount
            if 1 > self.cfg.graphnas.controller.discount > 0:
                rewards = discount(rewards, self.cfg.graphnas.controller.discount)

            reward_history.extend(rewards)
            entropy_history.extend(np_entropies)

            # moving average baseline
            if baseline is None:
                baseline = rewards
            else:
                decay = self.cfg.graphnas.controller.ema_baseline_decay
                baseline = decay * baseline + (1 - decay) * rewards

            adv = rewards - baseline
            history.append(adv)
            adv = scale(adv, scale_value=0.5)
            adv_history.extend(adv)

            adv = utl.get_variable(adv, self.cuda, requires_grad=False)
            # policy loss
            loss = -log_probs * adv
            if self.cfg.graphnas.controller.entropy_mode == 'regularizer':
                loss -= self.cfg.graphnas.controller.entropy_coeff * entropies

            loss = loss.sum()  # or loss.mean()

            # update
            self.controller_optim.zero_grad()
            loss.backward()

            if self.cfg.graphnas.controller.controller_grad_clip > 0:
                torch.nn.utils.clip_grad_norm(model.parameters(),
                                              self.cfg.graphnas.controller.controller_grad_clip)
            self.controller_optim.step()

            total_loss += utl.to_item(loss.data)

            self.controller_step += 1
            torch.cuda.empty_cache()

        print("*" * 35, "training controller over", "*" * 35)

    def evaluate(self, gnn):
        """
        Evaluate a structure on the validation set.
        """
        self.controller.eval()
        gnn = self.form_gnn_info(gnn)
        results = self.submodel_manager.retrain(gnn, format=self.cfg.graphnas.controller.format)
        if results:
            reward, scores = results
        else:
            return

        logger.info(f'eval | {gnn} | reward: {reward:8.2f} | scores: {scores:8.2f}')

    def derive_from_history(self):
        with open(self.cfg.data.type + "_" + self.cfg.graphnas.controller.search_mode + self.cfg.graphnas.controller.submanager_log_file, "a") as f:
            lines = f.readlines()

        results = []
        best_val_score = "0"
        for line in lines:
            actions = line[:line.index(";")]
            val_score = line.split(";")[-1]
            results.append((actions, val_score))
        results.sort(key=lambda x: x[-1], reverse=True)
        best_structure = ""
        best_score = 0
        for actions in results[:5]:
            actions = eval(actions[0])
            np.random.seed(123)
            torch.manual_seed(123)
            torch.cuda.manual_seed_all(123)
            val_scores_list = []
            for i in range(20):
                val_acc, test_acc = self.submodel_manager.evaluate(actions)
                val_scores_list.append(val_acc)

            tmp_score = np.mean(val_scores_list)
            if tmp_score > best_score:
                best_score = tmp_score
                best_structure = actions

        print("best structure:" + str(best_structure))
        # train from scratch to get the final score
        np.random.seed(123)
        torch.manual_seed(123)
        torch.cuda.manual_seed_all(123)
        test_scores_list = []
        for i in range(100):
            # manager.shuffle_data()
            val_acc, test_acc = self.submodel_manager.evaluate(best_structure)
            test_scores_list.append(test_acc)
        print(f"best results: {best_structure}: {np.mean(test_scores_list):.8f} +/- {np.std(test_scores_list)}")
        return best_structure

    # def derive(self, sample_num=None):
    #     """
    #     sample a serial of structures, and return the best structure.
    #     """
    #     if sample_num is None and self.cfg.derive_from_history:
    #         return self.derive_from_history()
    #     else:
    #         if sample_num is None:
    #             sample_num = self.cfg.derive_num_sample
    #
    #         gnn_list, _, entropies = self.controller.sample(sample_num, with_details=True)
    #
    #         max_R = 0
    #         best_actions = None
    #         filename = self.model_info_filename
    #         for action in gnn_list:
    #             gnn = self.form_gnn_info(action)
    #             reward = self.submodel_manager.test_with_param(gnn, format=self.cfg.graphnas.controller.format,
    #                                                            with_retrain=self.with_retrain)
    #
    #             if reward is None:  # cuda error hanppened
    #                 continue
    #             else:
    #                 results = reward[1]
    #
    #             if results > max_R:
    #                 max_R = results
    #                 best_actions = action
    #
    #         logger.info(f'derive |action:{best_actions} |max_R: {max_R:8.6f}')
    #         self.evaluate(best_actions)
    #         return best_actions

    @property
    def model_info_filename(self):
        return f"{self.cfg.data.type}_{self.cfg.graphnas.controller.search_mode}_{self.cfg.graphnas.controller.format}_results.txt"

    @property
    def controller_path(self):
        return f'{self.cfg.data.type}/controller_epoch{self.epoch}_step{self.controller_step}.pth'

    @property
    def controller_optimizer_path(self):
        return f'{self.cfg.data.type}/controller_epoch{self.epoch}_step{self.controller_step}_optimizer.pth'

    def get_saved_models_info(self):
        paths = glob.glob(os.path.join(self.cfg.data.type, '*.pth'))
        paths.sort()

        def get_numbers(items, delimiter, idx, replace_word, must_contain=''):
            return list(set([int(
                name.split(delimiter)[idx].replace(replace_word, ''))
                for name in items if must_contain in name]))

        basenames = [os.path.basename(path.rsplit('.', 1)[0]) for path in paths]
        epochs = get_numbers(basenames, '_', 1, 'epoch')
        shared_steps = get_numbers(basenames, '_', 2, 'step', 'shared')
        controller_steps = get_numbers(basenames, '_', 2, 'step', 'controller')

        epochs.sort()
        shared_steps.sort()
        controller_steps.sort()

        return epochs, shared_steps, controller_steps

    def save_model(self):

        torch.save(self.controller.state_dict(), self.controller_path)
        torch.save(self.controller_optim.state_dict(), self.controller_optimizer_path)

        logger.info(f'[*] SAVED: {self.controller_path}')

        epochs, shared_steps, controller_steps = self.get_saved_models_info()

        for epoch in epochs[:-self.cfg.graphnas.controller.max_save_num]:
            paths = glob.glob(
                os.path.join(self.cfg.data.type, f'*_epoch{epoch}_*.pth'))

            for path in paths:
                utl.remove_file(path)

    def load_model(self):
        epochs, shared_steps, controller_steps = self.get_saved_models_info()

        if len(epochs) == 0:
            logger.info(f'[!] No checkpoint found in {self.cfg.data.type}...')
            return

        self.epoch = self.start_epoch = max(epochs)
        self.controller_step = max(controller_steps)

        self.controller.load_state_dict(
            torch.load(self.controller_path))
        self.controller_optim.load_state_dict(
            torch.load(self.controller_optimizer_path))
        logger.info(f'[*] LOADED: {self.controller_path}')
