import logging
import copy
import os
import sys

import numpy as np
import pickle

from src.federatedscope.core.monitors.early_stopper import EarlyStopper
from src.federatedscope.core.message import Message
from src.federatedscope.core.communication import StandaloneCommManager, \
    gRPCCommManager
from src.federatedscope.core.auxiliaries.aggregator_builder import get_aggregator
from src.federatedscope.core.auxiliaries.sampler_builder import get_sampler
from src.federatedscope.core.auxiliaries.utils import merge_dict_of_results, \
    Timeout, merge_param_dict
from src.federatedscope.core.auxiliaries.trainer_builder import get_trainer
from src.federatedscope.core.secret_sharing import AdditiveSecretSharing
from src.federatedscope.core.workers.server import Server
from src.federatedscope.core.workers.client import Client
from src.federatedscope.core.workers.base_server import BaseServer
import torch
logger = logging.getLogger(__name__)


class AgnnsServer(Server):
    def _perform_federated_aggregation(self):
        """
        Perform federated aggregation and update the global model
        """
        train_msg_buffer = self.msg_buffer['train'][self.state]
        for model_idx in range(self.model_num):
            model = self.models[model_idx]
            aggregator = self.aggregators[model_idx]
            msg_list = list()
            staleness = list()

            for client_id in train_msg_buffer.keys():
                if self.model_num == 1:
                    msg_list.append(train_msg_buffer[client_id])
                else:
                    train_data_size, model_para_multiple = \
                        train_msg_buffer[client_id]
                    msg_list.append(
                        (train_data_size, model_para_multiple[model_idx]))

                # The staleness of the messages in train_msg_buffer
                # should be 0
                staleness.append((client_id, 0))

            for staled_message in self.staled_msg_buffer:
                state, client_id, content = staled_message
                if self.model_num == 1:
                    msg_list.append(content)
                else:
                    train_data_size, model_para_multiple = content
                    msg_list.append(
                        (train_data_size, model_para_multiple[model_idx]))

                staleness.append((client_id, self.state - state))

            # Trigger the monitor here (for training)


            self._monitor.calc_model_metric(self.model.state_dict(),
                                            msg_list,
                                            rnd=self.state)

            # Aggregate
            aggregated_num = len(msg_list)
            agg_info = {
                'client_feedback': msg_list,
                'recover_fun': self.recover_fun,
                'staleness': staleness,
            }
            # logger.info(f'The staleness is {staleness}')
            result = aggregator.aggregate(agg_info)
            # Due to lazy load, we merge two state dict
            merged_param = merge_param_dict(model.state_dict().copy(), result)
            model.load_state_dict(merged_param, strict=False)

        return aggregated_num




class AgnnsClient(Client):
    def callback_funcs_for_model_para(self, message: Message):
        """
        The handling function for receiving model parameters, \
        which triggers the local training process. \
        This handling function is widely used in various FL courses.

        Arguments:
            message: The received message
        """
        if 'ss' in message.msg_type:
            # A fragment of the shared secret
            state, content, timestamp = message.state, message.content, \
                                        message.timestamp
            self.msg_buffer['train'][state].append(content)

            if len(self.msg_buffer['train']
                   [state]) == self._cfg.federate.client_num:
                # Check whether the received fragments are enough
                model_list = self.msg_buffer['train'][state]
                sample_size, first_aggregate_model_para = model_list[0]
                single_model_case = True
                if isinstance(first_aggregate_model_para, list):
                    assert isinstance(first_aggregate_model_para[0], dict), \
                        "aggregate_model_para should a list of multiple " \
                        "state_dict for multiple models"
                    single_model_case = False
                else:
                    assert isinstance(first_aggregate_model_para, dict), \
                        "aggregate_model_para should " \
                        "a state_dict for single model case"
                    first_aggregate_model_para = [first_aggregate_model_para]
                    model_list = [[model] for model in model_list]

                for sub_model_idx, aggregate_single_model_para in enumerate(
                        first_aggregate_model_para):
                    for key in aggregate_single_model_para:
                        for i in range(1, len(model_list)):
                            aggregate_single_model_para[key] += model_list[i][
                                sub_model_idx][key]

                self.comm_manager.send(
                    Message(msg_type='model_para',
                            sender=self.ID,
                            receiver=[self.server_id],
                            state=self.state,
                            timestamp=timestamp,
                            content=(sample_size, first_aggregate_model_para[0]
                                     if single_model_case else
                                     first_aggregate_model_para)))

        else:
            round = message.state
            sender = message.sender
            timestamp = message.timestamp
            content = message.content
            # When clients share the local model, we must set strict=True to
            # ensure all the model params (which might be updated by other
            # clients in the previous local training process) are overwritten
            # and synchronized with the received model
            self.trainer.update(content,
                                strict=self._cfg.federate.share_local_model)
            self.state = round
            skip_train_isolated_or_global_mode = \
                self.early_stopper.early_stopped and \
                self._cfg.federate.method in ["local", "global"]
            if self.is_unseen_client or skip_train_isolated_or_global_mode:
                # for these cases (1) unseen client (2) isolated_global_mode,
                # we do not local train and upload local model
                sample_size, model_para_all, results= \
                    0, self.trainer.get_model_para(), {}
                if skip_train_isolated_or_global_mode:
                    logger.info(
                        f"[Local/Global mode] Client #{self.ID} has been "
                        f"early stopped, we will skip the local training")
                    self._monitor.local_converged()
            else:
                if self.early_stopper.early_stopped and \
                        self._monitor.local_convergence_round == 0:
                    logger.info(
                        f"[Normal FL Mode] Client #{self.ID} has been locally "
                        f"early stopped. "
                        f"The next FL update may result in negative effect")
                    self._monitor.local_converged()
                sample_size, model_para_all, results,last_model = self.trainer.train()
                if self._cfg.federate.share_local_model and not \
                        self._cfg.federate.online_aggr:
                    model_para_all = copy.deepcopy(model_para_all)
                train_log_res = self._monitor.format_eval_res(
                    results,
                    rnd=self.state,
                    role='Client #{}'.format(self.ID),
                    return_raw=True)
                logger.info(train_log_res)
                if self._cfg.wandb.use and self._cfg.wandb.client_train_info:
                    self._monitor.save_formatted_results(train_log_res,
                                                         save_file_name="")

            # Return the feedbacks to the server after local update
            if self._cfg.federate.use_ss:
                assert not self.is_unseen_client, \
                    "Un-support using secret sharing for unseen clients." \
                    "i.e., you set cfg.federate.use_ss=True and " \
                    "cfg.federate.unseen_clients_rate in (0, 1)"
                single_model_case = True
                if isinstance(model_para_all, list):
                    assert isinstance(model_para_all[0], dict), \
                        "model_para should a list of " \
                        "multiple state_dict for multiple models"
                    single_model_case = False
                else:
                    assert isinstance(model_para_all, dict), \
                        "model_para should a state_dict for single model case"
                    model_para_all = [model_para_all]
                model_para_list_all = []
                for model_para in model_para_all:
                    for key in model_para:
                        model_para[key] = model_para[key] * sample_size
                    model_para_list = self.ss_manager.secret_split(model_para)
                    model_para_list_all.append(model_para_list)
                    # print(model_para)
                    # print(self.ss_manager.secret_reconstruct(
                    # model_para_list))
                frame_idx = 0
                for neighbor in self.comm_manager.neighbors:
                    if neighbor != self.server_id:
                        content_frame = model_para_list_all[0][frame_idx] if \
                            single_model_case else \
                            [model_para_list[frame_idx] for model_para_list
                             in model_para_list_all]
                        self.comm_manager.send(
                            Message(msg_type='ss_model_para',
                                    sender=self.ID,
                                    receiver=[neighbor],
                                    state=self.state,
                                    timestamp=self._gen_timestamp(
                                        init_timestamp=timestamp,
                                        instance_number=sample_size),
                                    content=content_frame))
                        frame_idx += 1
                content_frame = model_para_list_all[0][frame_idx] if \
                    single_model_case else \
                    [model_para_list[frame_idx] for model_para_list in
                     model_para_list_all]
                self.msg_buffer['train'][self.state] = [(sample_size,
                                                         content_frame)]
            else:
                if self._cfg.asyn.use:
                    # Return the model delta when using asynchronous training
                    # protocol, because the staled updated might be discounted
                    # and cause that the sum of the aggregated weights might
                    # not be equal to 1
                    shared_model_para = self._calculate_model_delta(
                        init_model=content, updated_model=model_para_all)
                else:
                    shared_model_para = model_para_all

                self.comm_manager.send(
                    Message(msg_type='model_para',
                            sender=self.ID,
                            receiver=[sender],
                            state=self.state,
                            timestamp=self._gen_timestamp(
                                init_timestamp=timestamp,
                                instance_number=sample_size),
                            content=(sample_size, shared_model_para)))  #content=(sample_size, shared_model_para,last_model)






