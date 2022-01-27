"""
    1、focal loss
    2、所有参数可以加到config的args字典中(参考ccks)
    3、Negative Sampling
    BUG：
    2、model.load()加载模型参数维度出问题
    3、模型效果太差打印batch的前几个看看数据是否对齐
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
    train_data, valid_data, core_cust_id_size, prod_code_size = gp_csv_data(train_path='data/x_train_process.csv',
                                                                            test_path='data/x_test_process.csv',
                                                                            return_type='train'
                                                                            )
    dense_feature = ['year','month','day','d1','d2','d3','g8','k4','k6','k7','k8','k9']
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
    optimizer = optim.AdamW(CLS_model.parameters(), lr=0.001, weight_decay=0.01)
    trainer.training(
                        model = CLS_model, 
                        device = DEVICE,
                        epochs = 8,
                        train_dl = train_dl,
                        valid_dl = valid_dl,
                        criterion = BCEFocalLoss(),
                        optimizer = optimizer
                    )


if __name__ == '__main__':
        main()