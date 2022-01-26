"""
    1、focal loss
    2、id这一列没用,唯一客户id和产品id做embedding即可,加入id太容易过拟合
    BUG：
    1、plotting_utils()有bug,画不出图
    2、model.load()加载模型参数维度出问题
    3、outfile.to_csv('outfile.csv', index=False, encoding='gbk',float_format='%.3f')结果末尾要保留3位小数
"""
import warnings
import pandas as pd
warnings.filterwarnings("ignore")
import torch
import torch.nn as nn
import torch.optim as optim
from DataSet.data_process import (
                                   gp_csv_data,
                                   get_data_loader
                                 )
from Training.trainner import Trainner
from net.ClsModule import ClsModule


# DEVICE
USE_CUDA = torch.cuda.is_available()
DEVICE = torch.device("cuda:0" if USE_CUDA else "cpu")


def main():
    train_data, valid_data, test_data, core_cust_id_size, prod_code_size = gp_csv_data(train_path='data/x_train_process.csv',
                                                                                       test_path='data/x_test_process.csv',
                                                                                       rows=200000)
    train_dl = get_data_loader('train',train_data)
    valid_dl = get_data_loader('valid',valid_data)
    dense_feature = ['year','month','day','d1','d2','d3','g1','g2','g3',
                     'g4','g5','g6','g7','g8','k4','k6','k7','k8','k9']
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
                        criterion = nn.BCELoss(),
                        optimizer = optimizer
                    )


def predict():
    """
        model的id_size(embedding),core_cust_id_size(embedding),prod_code_size(embedding)
        三个embedding_input的维度需要和train中的维度对齐,否则会因为test和train的数据分布维度差异导致无法预测
    """
    csv_data, _, _, _ = gp_csv_data('data/x_test_process.csv')
    id_test_dl, core_cust_id_test_dl, prod_code_test_dl, dense_test_dl = get_test_data_loader(csv_data)
    dense_feature = ['year','month','day','d1','d2','d3','g1','g2','g3',
                     'g4','g5','g6','g7','g8','k4','k6','k7','k8','k9']
    model = ClsModule(
                        dense_feature_columns = dense_feature,
                        hidden_units = [1024, 512],
                        id_size = 21,
                        core_cust_id_size = 17,
                        prod_code_size = 6,
                     )
    checkpoint = torch.load('output/checkpoints/model_state.pt')
    model.load_state_dict(checkpoint['model_state'])
    # torch.no_grad()
    model.eval()
    output = model(
                    id_input = id_test_dl,
                    core_cust_id_input = core_cust_id_test_dl,
                    prod_code_input = prod_code_test_dl,
                    dense_input = dense_test_dl
                  )

    print(output[:30])
    return


def get_submission(result):
    test = pd.read_csv('data/x_test_process.csv', encoding='UTF-8')
    test['y'] = result
    result = test[['id','y']]

    result.to_csv('data/result.csv',index=False)


if __name__ == '__main__':
        main()
        # get_submission(result)
        # predict()