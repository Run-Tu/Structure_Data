import torch
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

class BCEFocalLoss(torch.nn.Module):
    """
        预测结果predict接近无线接近于0时log(predict)无限接近于负无穷,导致结果NaN
    """
    def __init__(self, gamma=2, alpha=0.5, reduction='mean'):
        super(BCEFocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction
    
    def forward(self, predict, target):
        min_pro = torch.tensor(0.001, dtype=torch.float32)
        predict = torch.maximum(predict, min_pro)
        loss = - self.alpha * (1 - predict) ** self.gamma * target * torch.log(predict) - (1 - self.alpha) * predict ** self.gamma * (1 - target) * torch.log(1 - predict)
        
        return torch.mean(loss)