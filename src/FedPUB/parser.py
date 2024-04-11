import argparse
from src.FedPUB.misc.utils import *


class Parser:
    def __init__(self):
        self.parser = argparse.ArgumentParser()
        self.set_arguments()

    def set_arguments(self):
        self.parser.add_argument('--gpu', type=str, default='2,3,4')  # cfg.gpu_list
        self.parser.add_argument('--seed', type=int, default=1234)

        self.parser.add_argument('--model', type=str, default=None)  # federate.method
        self.parser.add_argument('--dataset', type=str, default=None)  # data.type
        self.parser.add_argument('--mode', type=str, default=None, choices=['disjoint', 'overlapping'])  # data.mode
        self.parser.add_argument('--base-path', type=str, default='res')

        self.parser.add_argument('--n_workers', type=int, default=None)  # federate.client_num.n_workers
        self.parser.add_argument('--n-clients', type=int, default=None)  # federate.client_num
        self.parser.add_argument('--n-rnds', type=int, default=None)  # federate.total_round_num
        self.parser.add_argument('--n-eps', type=int, default=None)  # federate.n_eps
        self.parser.add_argument('--frac', type=float, default=None)  # federate.frac
        self.parser.add_argument('--n-dims', type=int, default=128)  # data
        self.parser.add_argument('--lr', type=float, default=None)  # train.optimizer.lr

        self.parser.add_argument('--laye-mask-one', action='store_true')  # federate.laye_mask_one = True
        self.parser.add_argument('--clsf-mask-one', action='store_true')  # federate.clsf_mask_one = True

        self.parser.add_argument('--agg-norm', type=str, default='exp', choices=['cosine', 'exp'])  # federate.agg_norm
        self.parser.add_argument('--norm-scale', type=float, default=10)  # federate.norm_scale
        self.parser.add_argument('--n-proxy', type=int, default=5)  # cfg.federate.n_proxy = 5

        self.parser.add_argument('--l1', type=float, default=1e-3) # cfg.federate.l1
        self.parser.add_argument('--loc-l2', type=float, default=1e-3) #cfg.federate.loc-l2

        self.parser.add_argument('--debug', action='store_true')

    def parse(self):
        args, unparsed = self.parser.parse_known_args()
        if len(unparsed) != 0:
            raise SystemExit('Unknown argument: {}'.format(unparsed))
        return args
