from __future__ import division
import numpy as np

import matplotlib.pyplot as plt

def accuracy(labels, predictions):
    # for i in range(len(labels)):
    #     if labels[i] ==
    # labels2 = np.array([])
    # print(labels)
    # labels = np.where(labels == 'True', 1, labels2)
    # labels = np.where(labels == 'False', 0, labels2)
    predictions = np.array(predictions).astype(np.int)
    labels = np.array(labels)
    labels = labels.reshape((-1,1))
    # print(predictions)
    # print(labels)
    # print(labels == predictions)

    return np.sum(labels == predictions) / len(labels)


def precision(labels, predictions):
    positive_predictions = (predictions == 1)

    labels = labels.reshape((-1,1))
    if positive_predictions.sum() == 0:
        return 1.0
    return ((labels == predictions) & positive_predictions).sum() / positive_predictions.sum()


def recall(labels, predictions):
    labels = np.array(labels)
    positive_label = (labels == 1)

    labels = labels.reshape((-1,1))
    if positive_label.sum() == 0:
        return 1.0
    return ((labels == predictions) & positive_label).sum() / positive_label.sum()


def specificity(labels, predictions):
    negative_label = np.array(labels == -1)

    labels = labels.reshape((-1,1))
    if negative_label.sum() == 0:
        return 1.0
    return ((labels == predictions) & negative_label).sum() / negative_label.sum()

def area_under_curve(labels, predictions):

    labels = labels.reshape((-1,1))
    cutoffs = np.sort(np.unique(np.append(predictions, [0, 1])), )
    area_under_curve = 0
    previous_true_prob = 1
    previous_false_prob = 1
    true_probs = []
    false_probs = []
    for x in cutoffs:
        pred = np.where(predictions > x, 1, -1)
        tpr = recall(labels, pred)
        fpr = 1 - specificity(labels, pred)
        area_under_curve += (previous_false_prob - fpr) * (tpr + previous_true_prob) * 0.5
        previous_true_prob = tpr
        previous_false_prob = fpr
        true_probs.append(tpr)
        false_probs.append(fpr)

    plt.style.use('ggplot')
    fig, axes = plt.subplots()
    axes.plot(false_probs, true_probs)
    axes.set_xlabel('False Positive Rate')
    axes.set_ylabel('True Positive Rate')
    fig.savefig('roc.pdf')

    return area_under_curve