import pandas as pd
import torch
from DataSet.dataset import (
                            idDataSet,
                            core_cust_idDataSet,
                            prod_codeDataSet,
                            )
from torch.utils.data import TensorDataset
from torch.utils.data.dataloader import DataLoader


def gp_csv_data(train_path, test_path, rows=None):
    """
        Get and process csv data
        get train_data, valid_data, test_data
        get id, core_cust_id, prod_code embed size
    """
    train_data = pd.read_csv(train_path, encoding='utf-8', nrows=rows)
    test_data = pd.read_csv(test_path, encoding='utf-8') # test_data得搞全量数据集
    data = train_data.append(test_data)
    # id_2_num
    id_2_num = {}
    for idx, id in enumerate(set(data['id'].values)):
        if id not in id_2_num.keys():
            id_2_num[id] = idx
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
    data['id'] = data['id'].map(id_2_num)
    data['core_cust_id'] = data['core_cust_id'].map(core_cust_id_2_num)
    data['prod_code'] = data['prod_code'].map(prod_code_2_num)
    # split train and test data
    train = data.iloc[:-567362]
    train_data = train.iloc[ : int(len(train)*0.7)]
    valid_data = train.iloc[int(len(train)*0.7) : ]
    test_data = data.iloc[-567362:].drop('y', axis=1)
    # calculate embedding matrix dim
    id_size = len(id_2_num)
    core_cust_id_size = len(core_cust_id_2_num)
    prod_code_size = len(prod_code_2_num)  

    return train_data, valid_data, test_data, id_size, core_cust_id_size, prod_code_size


def feature_enginering(df):
    pass


def get_data_loader(data_type:str, csv_data):
    """
        这个功能还可以优化
        通过':'设定方法的形参类型
    """
    assert data_type in ['train','valid','test']
    id_feature = 'id'
    core_cust_id_feature = 'core_cust_id'
    prod_code_feature = 'prod_code'
    dense_feature = ['year','month','day','d1','d2','d3','g1','g2','g3',
                     'g4','g5','g6','g7','g8','k4','k6','k7','k8','k9']
    # DataLoader的shuffle
    dataloader_params = {
                        'batch_size':1024,
                        'shuffle':False,
                        'drop_last':True,
                        'num_workers':2
                        }

    if data_type in ['train','valid']:
        # get torch dataset
        target = 'y'
        id_dataset = idDataSet(csv_data, id_feature)
        core_cust_id_dataset = core_cust_idDataSet(csv_data, core_cust_id_feature)
        prod_code_dataset = prod_codeDataSet(csv_data, prod_code_feature)
        dense_dataset = TensorDataset(
                                    torch.tensor(csv_data[dense_feature].values).float(), 
                                    torch.tensor(csv_data[target].values).float()
                                    )       
        # get torch dataloader
        id_dl = DataLoader(id_dataset, **dataloader_params)
        core_cust_id_dl = DataLoader(core_cust_id_dataset, **dataloader_params)
        prod_code_dl = DataLoader(prod_code_dataset, **dataloader_params)
        dense_dl = DataLoader(dense_dataset, **dataloader_params)
    else:
        # get torch dataset
        id_dataset = idDataSet(csv_data, id_feature)
        core_cust_id_dataset = core_cust_idDataSet(csv_data, core_cust_id_feature)
        prod_code_dataset = prod_codeDataSet(csv_data, prod_code_feature)
        dense_dataset = TensorDataset(
                                    torch.tensor(csv_data[dense_feature].values).float(), 
                                    )       
        # get torch dataloader
        id_dl = DataLoader(id_dataset, **dataloader_params)
        core_cust_id_dl = DataLoader(core_cust_id_dataset, **dataloader_params)
        prod_code_dl = DataLoader(prod_code_dataset, **dataloader_params)
        dense_dl = DataLoader(dense_dataset, **dataloader_params)

    return id_dl, core_cust_id_dl, prod_code_dl, dense_dl

