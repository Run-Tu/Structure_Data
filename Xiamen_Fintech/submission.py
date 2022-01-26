import warnings
import time
import torch
import pandas as pd
warnings.filterwarnings("ignore")
import torch
from DataSet.data_process import (
                                   gp_csv_data,
                                   get_data_loader
                                 )
from net.ClsModule import ClsModule

# DEVICE
USE_CUDA = torch.cuda.is_available()
DEVICE = torch.device("cuda:0" if USE_CUDA else "cpu")
TRAIN_TIME = time.strftime("%Y-%m-%d", time.localtime())


def predict(device, TRAIN_TIME):
    """
        model的id_size(embedding),core_cust_id_size(embedding),prod_code_size(embedding)
        三个embedding_input的维度需要和train中的维度对齐,否则会因为test和train的数据分布维度差异导致无法预测
    """
    test_data, core_cust_id_size, prod_code_size = gp_csv_data(train_path='data/x_train_process.csv',
                                                               test_path='data/x_test_process.csv',
                                                               return_type='test')
    test_dl = get_data_loader('test', test_data)
    dense_feature = ['year','month','day','d1','d2','d3','g1','g2','g3',
                     'g4','g5','g6','g7','g8','k4','k6','k7','k8','k9']

    checkpoint = torch.load(f'output/checkpoints/{TRAIN_TIME}_model_state.pt')
    CLS_model = ClsModule(
                          dense_feature_columns = dense_feature,
                          hidden_units = [1024, 512],
                          core_cust_id_size = core_cust_id_size,
                          prod_code_size = prod_code_size
                         ).to(device)
    CLS_model.load_state_dict(checkpoint['model_state'])
    # 切换测试模式
    torch.no_grad() # detach()和no_grad()作用一样，detach()面向变量，no_grad()面向整体
    CLS_model.eval()
    result = []
    for core_cust_id_batch, prod_code_batch, dense_batch in test_dl:
            # transfer data type
            core_cust_id_batch = core_cust_id_batch.long().to(device)
            print("core_cust_id_batch is", core_cust_id_batch)
            prod_code_batch = prod_code_batch.long().to(device)
            print("prod_code_batch is", prod_code_batch)
            print("dense_batch is", dense_batch)
            dense_batch = dense_batch.float().to(device)
            
            # calculate y_pred
            output = CLS_model(
                                core_cust_id_input = core_cust_id_batch,
                                prod_code_input = prod_code_batch,
                                dense_input = dense_batch
                              )
            result.extend(torch.squeeze(output))


    return result


def get_submission(result):
    test = pd.read_csv('data/x_test_process.csv', encoding='UTF-8')
    test['y'] = result
    result = test[['id','y']]
    # 保存结果小数点后6位
    result.to_csv('data/result.csv', encoding='UTF-8', index=False, float_format='%.6f')


if __name__ == '__main__':
    predict(DEVICE, TRAIN_TIME)