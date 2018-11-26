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

def f1_values(y_train, y_validation_train):
	tp = 0
	fp = 0
	tn = 0
	fn = 0
	for i in range(len(y_train)):
		if y_train[i,1] == 1:
			if y_validation_train[i] == 1:
				tp += 1
			else:
				fn += 1
		else:
			if y_validation_train[i] == 1:
				fp += 1
			else:
				tn += 1
	return tp, tn, fp, fn
				



