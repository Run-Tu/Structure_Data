"""
    看一下soul的loss横坐标是epoch还是iter？
"""
import matplotlib.pyplot as plt
from sklearn.metrics import precision_score, recall_score


def calculate_F2_score(y_pred, y_true):
    """
        可以定义成utils
    """
    y_pred = [1 if i>=0.5 else 0 for i in y_pred]
    precision = precision_score(y_pred, y_true)
    recall = recall_score(y_pred, y_true)
    F2_score = 5 * precision * recall / (4 * precision + recall)

    return F2_score