"""
    将不同代码下的日志保存到不同的文件中是否需要Handler？怎么处理？
"""
import math
import torch
import torch.nn as nn


class ClsModule(nn.Module):
    """
        core_cust_id,prod_code需要2个embedding,不包括id
        经验embedding_dim = log2(feature_size)
        经验embedding_init?
        将3个embedding后的向量和连续型特征cat做nn.Linear()接sigmoid
        3个id分别做embedding
    """
    def __init__(self, dense_feature_columns, hidden_units, core_cust_id_size=264055, prod_code_size=129, 
                 dropout=False , dropout_rate=0.2):
        """
            core_cust_id,prod_code默认全量数据大小
            训练时会自动计算core_cust_id_size和prod_code_size
            Args: 
                dense_feature(list) -> dense features list 
                core_cust_id_size -> numbers of core_cust_id_vocab
                prod_code_size -> numbers of prod_code_size_vocab
        """
        super(ClsModule, self).__init__()
        self.dense_feature_dim = len(dense_feature_columns)
        # id,core_cust_id,prod_code size
        self.core_cust_id_size = core_cust_id_size
        self.prod_code_size = prod_code_size
        # embedding_dim
        self.core_cust_id_embed_dim = int(math.log2(self.core_cust_id_size))
        self.prod_code_embed_dim = int(math.log2(self.prod_code_size))
        # embedding层
        # embedding层indexError:index out of range in self 参考：https://blog.csdn.net/qq_40311018/article/details/115586527
        self.embed_layer = nn.ModuleDict({
            "embed_core_cust_id" : nn.Embedding(num_embeddings=self.core_cust_id_size, embedding_dim=self.core_cust_id_embed_dim),
            "embed_prod_code" : nn.Embedding(num_embeddings=self.prod_code_size, embedding_dim=self.prod_code_embed_dim)
            })
        # dnn层
        self.dnn_input_size = self.core_cust_id_embed_dim + self.prod_code_embed_dim + self.dense_feature_dim
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


    def forward(self, core_cust_id_input, prod_code_input, dense_input):
        """
            id,core_cust_id,prod_code三个sparse需要接入3个embedding层
        """
        core_cust_id_embed_output = self.embed_layer["embed_core_cust_id"](core_cust_id_input)
        prod_code_embed_output = self.embed_layer["embed_prod_code"](prod_code_input)
        dnn_input = torch.cat([
                               core_cust_id_embed_output,
                               prod_code_embed_output,
                               dense_input
                               ], axis=1)
        dnn_output = self.dnn_network(dnn_input)
        model_output = self.final_linear(dnn_output)
        model_output = torch.sigmoid(model_output)
        
        return model_output
        

