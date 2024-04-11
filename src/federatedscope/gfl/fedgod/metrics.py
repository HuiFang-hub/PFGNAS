# -*- coding: utf-8 -*-
# @Time    : 12/04/2023 20:17
# @Function:
from sklearn.metrics import roc_auc_score, average_precision_score, ndcg_score
from sklearn.metrics import accuracy_score,recall_score,roc_curve,auc,f1_score
def result_auc(labels, pred):
    # roc_auc = roc_auc_score(y_true=labels, y_score=pred)
    fpr, tpr, thresholds = roc_curve(y_true=labels, y_score=pred)
    roc_auc = auc(fpr, tpr)
    prec_5 = accuracy_score(labels, [1 if p >= sorted(pred, reverse=True)[:5][-1] else 0 for p in pred])
    prec_10 = accuracy_score(labels, [1 if p >= sorted(pred, reverse=True)[:10][-1] else 0 for p in pred])
    prec_20 = accuracy_score(labels, [1 if p >= sorted(pred, reverse=True)[:20][-1] else 0 for p in pred])
    return roc_auc, fpr, tpr, prec_5, prec_10, prec_20

def result_acc_rec_f1(labels, pred):
    # outlier detection is a binary classification problem
    acc = accuracy_score(y_true=labels, y_pred=pred)

    recall = recall_score(y_true=labels, y_pred=pred)

    f1 = f1_score(y_true=labels, y_pred=pred)
    return acc,recall,f1

def calculate_macro_recall(labels, predictions):
    macro_recall = recall_score(labels, predictions, average='macro')

    sample_weights = [0.3 if num == 0 else 0.7 for num in predictions]
    # 计算召回率（recall score）并考虑样本权重
    recall_weight = recall_score(labels, predictions, pos_label=1, average='binary', sample_weight=sample_weights)
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

