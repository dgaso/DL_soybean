# -*- coding: utf-8 -*-
"""
Created on Sun Aug  8 13:01:29 2021

@author: lauracue
"""
import numpy as np
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score

def evaluate_metrics(pred, gt, val = 0):
    accu_criteria = dict()
    if type(pred).__module__ != np.__name__:
        pred = pred.data.cpu().numpy()
    if type(gt).__module__ != np.__name__:
        gt = gt.data.cpu().numpy()

    c = pred.shape[1]

    pred = np.argmax(pred,axis=1)

    mask = np.where(gt>0)
    gt = gt[mask][:]
    pred = pred[mask][:]+1
    accuracy = accuracy_score(gt, pred)*100
    accu_criteria["Accuracy"] = np.round(accuracy,2)
    
    f1 = f1_score(gt, pred, average=None, zero_division=True)
    pre = precision_score(gt, pred, average=None, zero_division=True)
    rec = recall_score(gt, pred, average=None, zero_division=True)


    accu_criteria["avgF1"] = np.round(np.sum(f1)*100/c,2)
    accu_criteria["avgPre"] = np.round(np.sum(pre)*100/c,2)
    accu_criteria["avgRec"] = np.round(np.sum(rec)*100/c,2)
    # accu_criteria["F1"] = np.round(np.array(f1)*100,2)
    # accu_criteria["Pre"] = np.round(np.array(pre)*100,2)
    # accu_criteria["Rec"] = np.round(np.array(rec)*100,2)

    return accu_criteria



def evaluate_metrics_pred(pred, gt, val = 0):
    accu_criteria = dict()

    mask = np.where(gt>val)
    gt = gt[mask][:]
    pred = pred[mask][:]
    accuracy = accuracy_score(gt, pred)*100
    accu_criteria["Accuracy"] = np.round(accuracy,2)
    
    f1 = f1_score(gt, pred, average=None, zero_division=True)
    pre = precision_score(gt, pred, average=None, zero_division=True)
    rec = recall_score(gt, pred, average=None, zero_division=True)

    c = np.unique(gt)
    accu_criteria["avgF1"] = np.round(np.sum(f1)*100/c,2)
    accu_criteria["avgPre"] = np.round(np.sum(pre)*100/c,2)
    accu_criteria["avgRec"] = np.round(np.sum(rec)*100/c,2)
    accu_criteria["F1"] = np.round(np.array(f1)*100,2)
    accu_criteria["Pre"] = np.round(np.array(pre)*100,2)
    accu_criteria["Rec"] = np.round(np.array(rec)*100,2)

    return accu_criteria
