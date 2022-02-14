from torch.utils.data import Dataset


class core_cust_idDataSet(Dataset):
    """
        create core_cust_id column embedding input dataset
    """
    def __init__(self, csv_file, feature):
        super(core_cust_idDataSet, self).__init__()
        self.csv_file = csv_file
        self.core_cust_id_feature = feature
        self.core_cust_id_data = self.csv_file[self.core_cust_id_feature].values
    

    def __len__(self):

        return len(self.core_cust_id_data)
    

    def __getitem__(self, idx):

        return self.core_cust_id_data[idx]


class prod_codeDataSet(Dataset):
    """
        create prod_code column embedding input dataset
    """
    def __init__(self, csv_file, feature):
        super(prod_codeDataSet, self).__init__()
        self.csv_file = csv_file
        self.prod_code_feature = feature
        self.prod_code_data = self.csv_file[self.prod_code_feature].values
    

    def __len__(self):

        return len(self.prod_code_data)
    

    def __getitem__(self, idx):

        return self.prod_code_data[idx]

