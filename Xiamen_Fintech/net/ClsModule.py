"""
    将不同代码下的日志保存到不同的文件中是否需要Handler？怎么处理？
"""
import os
import math
import datetime
import logging
import torch
import torch.nn as nn


# # 日志模块
# TODAY = datetime.date.today()
# LOG_FORMAT = "%(asctime)s - %(levelname)s - %(message)s"
# DATE_FORMAT = "%Y/%m/%d %H:%M:%S %p"
# LOG_DIR = f'output/log/'
# if not os.path.exists(LOG_DIR):
#     os.makedirs(LOG_DIR)
# logging.basicConfig(
#                     filename=f"./output/log/{TODAY}.log", 
#                     level=logging.DEBUG, 
#                     format=LOG_FORMAT, 
#                     datefmt=DATE_FORMAT
#                    )


class ClsModule(nn.Module):
    """
        id,core_cust_id,prod_code需要3个embedding
        经验embedding_dim = log2(feature_size)
        经验embedding_init?
        将3个embedding后的向量和连续型特征cat做nn.Linear()接sigmoid
        3个id分别做embedding
    """
    def __init__(self, dense_feature_columns, hidden_units, id_size=2953495, core_cust_id_size=264055, prod_code_size=129, 
                 dropout=False , dropout_rate=0.2):
        """
            id_size,core_cust_id,prod_code默认全量数据大小
            Args: 
                dense_feature(list) -> dense features list 
                id_size(int) -> numbers of id_vocab
                core_cust_id_size -> numbers of core_cust_id_vocab
                prod_code_size -> numbers of prod_code_size_vocab
        """
        super(ClsModule, self).__init__()
        self.dense_feature_dim = len(dense_feature_columns)
        # id,core_cust_id,prod_code size
        self.id_size = id_size
        self.core_cust_id_size = core_cust_id_size
        self.prod_code_size = prod_code_size
        # embedding_dim
        self.id_embed_dim = int(math.log2(self.id_size))
        self.core_cust_id_embed_dim = int(math.log2(self.core_cust_id_size))
        self.prod_code_embed_dim = int(math.log2(self.prod_code_size))
        # embedding层
        # embedding层indexError:index out of range in self 参考：https://blog.csdn.net/qq_40311018/article/details/115586527
        self.embed_layer = nn.ModuleDict({
            "embed_id" : nn.Embedding(num_embeddings=self.id_size, embedding_dim=self.id_embed_dim),
            "embed_core_cust_id" : nn.Embedding(num_embeddings=self.core_cust_id_size, embedding_dim=self.core_cust_id_embed_dim),
            "embed_prod_code" : nn.Embedding(num_embeddings=self.prod_code_size, embedding_dim=self.prod_code_embed_dim)
            })
        # dnn层
        self.dnn_input_size = self.id_embed_dim + self.core_cust_id_embed_dim + self.prod_code_embed_dim + self.dense_feature_dim
        if dropout:
            self.dropout_rate = dropout_rate
            self.dnn_network = nn.Sequential(
                            nn.Linear(self.dnn_input_size, hidden_units[0]),
                            nn.Dropout(self.dropout_rate),
                            nn.ReLU(),
                            nn.Linear(hidden_units[0], hidden_units[1]),
                            nn.Dropout(self.dropout_rate),
                            nn.ReLU()
                )
        else:
            self.dnn_network = nn.Sequential(
                            nn.Linear(self.dnn_input_size, hidden_units[0]),
                            nn.ReLU(),
                            nn.Linear(hidden_units[0], hidden_units[1]),
                            nn.ReLU()
                )
        self.final_linear = nn.Linear(hidden_units[-1], 1)


    def forward(self, id_input, core_cust_id_input, prod_code_input, dense_input):
        """
            id,core_cust_id,prod_code三个sparse需要接入3个embedding层
        """
        id_embed_output = self.embed_layer["embed_id"](id_input)
        core_cust_id_embed_output = self.embed_layer["embed_core_cust_id"](core_cust_id_input)
        prod_code_embed_output = self.embed_layer["embed_prod_code"](prod_code_input)
        # get dnn layer input
        # torch.cat((A,B), axis=0)
        # logging.info("id_embed_output is ", id_embed_output)
        # logging.info("core_cust_id_embed_output is ", core_cust_id_embed_output)
        # logging.info("prod_code_embed_output is ", prod_code_embed_output)
        # logging.info("dense_input is ", dense_input)
        dnn_input = torch.cat([id_embed_output, 
                               core_cust_id_embed_output,
                               prod_code_embed_output,
                               dense_input], axis=1)
        # logging.info("dnn_input is ", dnn_input)
        dnn_output = self.dnn_network(dnn_input)
        # logging.info("dnn_input is ", dnn_output)
        model_output = self.final_linear(dnn_output)
        model_output = torch.sigmoid(model_output)
        
        return model_output
        

