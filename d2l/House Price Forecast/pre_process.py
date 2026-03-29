import pandas as pd
import torch
from pathlib import Path
import os

class PreProcessor():
    def __init__(self,train_data:Path|str,test_data:Path|str):
        self.train_data=pd.read_csv(train_data)
        self.test_data=pd.read_csv(test_data)
    

    def concat_features(self):
        """移除编号列和价格列，合并训练集和测试集的特征,方便后续处理"""
        return pd.concat((self.train_data.iloc[:,1:-1],self.test_data.iloc[:,1:]))
    
    def preprocess(self):
        """对训练集和测试集进行预处理"""
        all_features=self.concat_features()
        # 找到所有数值型特征的属性索引名称
        numeric_features=all_features.dtypes[all_features.dtypes!='object'].index

        # 对数值型特征进行归一化处理
        all_features[numeric_features]=all_features[numeric_features].apply(
            lambda x:(x-x.mean())/(x.std())
        )

        # 填充缺失值为0
        all_features[numeric_features]=all_features[numeric_features].fillna(0)

        # 独热编码+缺失值编码. “Dummy_na=True”将“na”（缺失值）视为有效的特征值，并为其创建指示符特征
        #关于为什么不直接将这些非数值的特征转换为数值型特征，而要将其转换为指示符特征，是因为这些特征的取值是离散的，不能直接进行数值归一化处理。
        # 而指示符特征作为独特编码存在，可以被模型直接使用，而不需要进行数值归一化处理。
        all_features = pd.get_dummies(all_features, dummy_na=True)
        all_features=all_features.astype('float32')

        #分割数据集
        n_train = self.train_data.shape[0]
        train_features = torch.tensor(all_features[:n_train].values, dtype=torch.float32)
        test_features = torch.tensor(all_features[n_train:].values, dtype=torch.float32)
        train_labels = torch.tensor(self.train_data.SalePrice.values.reshape(-1, 1), dtype=torch.float32)
       
        return train_features,test_features,train_labels
    
    def __call__(self):
        train_features,test_features,train_labels=self.preprocess()
        return train_features,test_features,train_labels

if __name__ == '__main__':
    dir_path=Path(__file__).resolve().parent
    pre_processor=PreProcessor(os.path.join(dir_path,'data/kaggle_house_pred_train.csv'),os.path.join(dir_path,'data/kaggle_house_pred_test.csv'))
    train_features,test_features,train_labels=pre_processor()

    print(train_features.shape,test_features.shape,train_labels.shape)