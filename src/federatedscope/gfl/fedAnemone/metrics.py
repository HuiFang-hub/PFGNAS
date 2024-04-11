# -*- coding: utf-8 -*-
# @Time    : 12/04/2023 20:17
# @Function:
from sklearn.metrics import roc_auc_score, average_precision_score, ndcg_score
from sklearn.metrics import accuracy_score,recall_score,roc_curve,auc,f1_score
import numpy as np
def result_auc(labels, pred):
    # roc_auc = roc_auc_score(y_true=labels, y_score=pred)
    fpr, tpr, thresholds = roc_curve(y_true=labels, y_score=pred)
    fpr =  np.around(fpr,decimals=5).tolist()
    tpr = np.around(tpr,decimals=5).tolist()
    roc_auc = round(auc(fpr, tpr),5)
    labels = labels.numpy()
    prec_5 = round(precision_at_k(labels, pred, k=5),5)
    prec_10 = round(precision_at_k(labels, pred, k=10),5)
    prec_20 =round(precision_at_k(labels, pred, k=20),5)
    return roc_auc, fpr, tpr, prec_5, prec_10, prec_20

def result_acc_rec_f1(labels, pred):
    # outlier detection is a binary classification problem
    acc = round(accuracy_score(y_true=labels, y_pred=pred),5)
    recall = round(recall_score(y_true=labels, y_pred=pred),5)
    f1 = round(f1_score(y_true=labels, y_pred=pred),5)
    return acc,recall,f1

def calculate_macro_recall(labels, predictions):
    macro_recall = round(recall_score(labels, predictions, average='macro'),5)

    sample_weights = [0.3 if num == 0 else 0.7 for num in predictions]
    # 计算召回率（recall score）并考虑样本权重
    recall_weight = round(recall_score(labels, predictions, pos_label=1, average='binary', sample_weight=sample_weights),5)
    # total_recall = 0.0
    # num_classes = len(labels)
    #
    # for label in labels:
    #     true_positives = sum(1 for p, t in zip(predictions, labels) if p == t == label)
    #     actual_positives = sum(1 for t in labels if t == label)
    #
    #     if actual_positives > 0:
    #         recall = true_positives / actual_positives
    #         total_recall += recall
    #
    # macro_recall = total_recall / num_classes
    return macro_recall,recall_weight

def precision_at_k(y_true, y_pred, k=5):
    sorted_indices = np.argsort(y_pred)[::-1]  # 按预测分数降序排列的索引
    top_k_indices = sorted_indices[:k]  # 取前k个索引
    top_k_labels = y_true[top_k_indices]  # 前k个索引对应的真实标签
    precision = round(np.sum(top_k_labels) / k ,5) # 计算精确度
    return precision