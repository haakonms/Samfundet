import numpy as np 


def precision(tp, fp):
    return tp / (tp + fp)


def recall(tp, fn):
    return tp / (tp + fn)


def accuracy(tp, fp, tn, fn):
    return (tp + tn) / (tp + tn + fp + fn)


def f1_score(tp, fp, fn):
    pre = precision(tp, fp)
    rec = recall(tp, fn)
    return (2.0 * pre * rec)/(pre + rec)

