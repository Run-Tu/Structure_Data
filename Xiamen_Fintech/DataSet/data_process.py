import pandas as pd
import torch
from DataSet.dataset import (
                            core_cust_idDataSet,
                            prod_codeDataSet,
                            )
from torch.utils.data import TensorDataset
from torch.utils.data.dataloader import DataLoader


def gp_csv_data(train_path, test_path, return_type:str, rows=None):
    """
        Get and process csv data
        get train_data, valid_data, test_data
        get id, core_cust_id, prod_code embed size
    """
    assert return_type in ['train','test']
    train_data = pd.read_csv(train_path, encoding='utf-8', nrows=rows)
    test_data = pd.read_csv(test_path, encoding='utf-8') # test_data得搞全量数据集
    data = train_data.append(test_data)
    data = data.drop('id', axis=1)
    # core_cust_id_2_num
    core_cust_id_2_num = {}
    for idx, core_cust_id in enumerate(set(data['core_cust_id'].values)):
        if core_cust_id not in core_cust_id_2_num.keys():
            core_cust_id_2_num[core_cust_id] = idx
    # prod_code_2_num
    prod_code_2_num = {}
    for idx, prod_code in enumerate(set(data['prod_code'].values)):
        if prod_code not in prod_code_2_num.keys():
            prod_code_2_num[prod_code] = idx
    # label emcoder
    data['core_cust_id'] = data['core_cust_id'].map(core_cust_id_2_num)
    data['prod_code'] = data['prod_code'].map(prod_code_2_num)
    # feature enginering
    def add_feature(df):
        """
            特征工程,负采样参考：https://zhuanlan.zhihu.com/p/387378387
            1、以理财产品为准,计算每个理财产品出现的次数
            2、以客户为准,计算每个客户购买的次数
        """
        df['prod_code_counts'] = df['prod_code'].map(df['prod_code'].value_counts())
        df['core_cust_id_counts'] = df['core_cust_id'].map(df['core_cust_id'].value_counts())

        return df
    data = add_feature(data)
    # split train and test data
    train = data.iloc[:-567362]
    train_data = train.iloc[ : int(len(train)*0.7)]
    valid_data = train.iloc[int(len(train)*0.7) : ]
    test_data = data.iloc[-567362:].drop('y', axis=1)
    # calculate embedding matrix dim
    core_cust_id_size = len(core_cust_id_2_num)
    prod_code_size = len(prod_code_2_num)  
    if return_type=='train':

        return train_data, valid_data, core_cust_id_size, prod_code_size
    else:

        return test_data, core_cust_id_size, prod_code_size


def get_data_loader(data_type:str, csv_data, dense_feature):
    assert data_type in ['train','valid','test']
    core_cust_id_feature = 'core_cust_id'
    prod_code_feature = 'prod_code'
    dense_feature = dense_feature
    # DataLoader的shuffle
    train_params = {
                        'batch_size':1024,
                        'shuffle':True,
                        'drop_last':True,
                        'num_workers':2
                        }
    test_params =  {
                        'batch_size':1024,
                        'shuffle':False,
                        'drop_last':False,
                        'num_workers':2
                        }
        # get common core_cust_id dataset and prod_code dataset
    core_cust_id_dataset = core_cust_idDataSet(csv_data, core_cust_id_feature)
    prod_code_dataset = prod_codeDataSet(csv_data, prod_code_feature) 
    
    if data_type in ['train','valid']:
        params = train_params
        # get dense part dataset
        target = 'y'
        dense_dataset = TensorDataset(
                                    torch.tensor(csv_data[dense_feature].values).float(), 
                                    torch.tensor(csv_data[target].values).float()
                                    )  
        dataset = list(zip(core_cust_id_dataset, prod_code_dataset, dense_dataset)) # 这里不变成list无法传入dataloader
        train_dataloader = DataLoader(dataset, **params)

        return train_dataloader

    else:
        params = test_params
        # get dense part dataset
        dense_dataset = TensorDataset(
                                    torch.tensor(csv_data[dense_feature].values).float()
                                    )      
        dataset = list(zip(core_cust_id_dataset, prod_code_dataset, dense_dataset)) # 这里不变成list无法传入dataloader
        test_dataloader = DataLoader(dataset, **params)
        return test_dataloader

