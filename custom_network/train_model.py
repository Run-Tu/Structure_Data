"""
    1、focal loss
    2、所有参数可以加到config的args字典中(参考ccks)
    3、Negative Sampling
    4、断点训练
    # 2022/2/7
    先提交一版修改过的代码，看一下分数效果
    改进思路：
        1、通过欠采样让正负样本平衡，训练一次大概在5W样本左右
        2、可以通过分组欠采样，一组数据训练一个模型，训练多个模型然后ensemble
"""
import warnings
warnings.filterwarnings("ignore")
import torch
import torch.optim as optim
from DataSet.data_process import (
                                   gp_csv_data,
                                   get_data_loader
                                 )
from Training.trainner import Trainner
from net.ClsModule import ClsModule
from utils.result_utils import BCEFocalLoss


# DEVICE
USE_CUDA = torch.cuda.is_available()
DEVICE = torch.device("cuda:0" if USE_CUDA else "cpu")


def main():
    """
        dense_feature需要自动处理,将id_feature剔除
    """
    train_data, valid_data, core_cust_id_size, prod_code_size = gp_csv_data(train_path='data/x_train_process.csv',
                                                                            test_path='data/x_test_B_process.csv',
                                                                            return_type='train',
                                                                            )
    dense_feature = ['year','month','day','prod_code_counts','core_cust_id_counts']
    train_dl = get_data_loader('train',train_data, dense_feature)
    valid_dl = get_data_loader('valid',valid_data, dense_feature)
    CLS_model = ClsModule(
                          dense_feature_columns = dense_feature,
                          hidden_units = [1024, 512],
                          core_cust_id_size = core_cust_id_size,
                          prod_code_size = prod_code_size
                         ).to(DEVICE)
    trainer = Trainner()
    # pytorch中一个模型(torch.nn.module)的可学习参数(权重和偏置值)是包含在模型参数(model.parameters())中的
    optimizer = optim.AdamW(CLS_model.parameters(), lr=5e-4, weight_decay=0.01)
    trainer.training(
                        model = CLS_model, 
                        device = DEVICE,
                        epochs = 2,
                        train_dl = train_dl,
                        valid_dl = valid_dl,
                        criterion = BCEFocalLoss(),
                        optimizer = optimizer
                    )


if __name__ == '__main__':
        main()