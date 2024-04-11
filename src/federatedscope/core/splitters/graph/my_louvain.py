# -*- coding: utf-8 -*-
# @Time    : 08/06/2023 09:57
# @Function:
import os
import sys
import torch
from torch_geometric.datasets import KarateClub
from torch_geometric.utils import to_networkx
import networkx as nx
from util.kmeans_gpu import kmeans

class my_clustering:
    def __init__(self, data,cluster_num):
        self.feature=data.x
        self.cluster_num = cluster_num

    def run(self):
        predict_labels, dis, initial = kmeans(X=self.feature, num_clusters=self.cluster_num, distance="euclidean", device="cuda")
        return  predict_labels
