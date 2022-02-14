import os
import torch
import datetime
import logging


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


def get_logging():
    # 日志模块
    TODAY = datetime.date.today()
    LOG_FORMAT = "%(asctime)s - %(levelname)s - %(message)s"
    DATE_FORMAT = "%Y/%m/%d %H:%M:%S %p"
    LOG_DIR = f'output/log/'
    if not os.path.exists(LOG_DIR):
        os.makedirs(LOG_DIR)
    logging.basicConfig(
                        filename=f"./output/log/{TODAY}.log", 
                        level=logging.DEBUG, 
                        format=LOG_FORMAT, 
                        datefmt=DATE_FORMAT
                    )
    
    return logging
