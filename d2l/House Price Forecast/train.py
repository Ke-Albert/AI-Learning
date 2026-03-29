import time
import torch
from torch import nn
from pathlib import Path
import os
from torch.utils.data import DataLoader, Dataset
import matplotlib.pyplot as plt

from pre_process import PreProcessor

def get_net(in_features):
    net=nn.Sequential(nn.Linear(in_features,512),nn.ReLU(),nn.Dropout(0.2),
                      nn.BatchNorm1d(512),nn.Linear(512,256),nn.ReLU(),nn.Dropout(0.5),
                      nn.BatchNorm1d(256),nn.Linear(256,64),nn.ReLU(),nn.Dropout(0.6),
                      nn.BatchNorm1d(64),nn.Linear(64,1))
    # net=nn.Sequential(nn.Linear(in_features,512),nn.ReLU(),
    #                   nn.BatchNorm1d(512),nn.Linear(512,1))

    def _weight_init(m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_normal_(m.weight)
        elif isinstance(m, nn.BatchNorm1d):
            nn.init.zeros_(m.bias)
            nn.init.ones_(m.weight)
    net.apply(_weight_init)
    return net

def log_rmse(net, features, labels):
    # 为了在取对数时进一步稳定该值，将小于1的值设置为1
    clipped_preds = torch.clamp(net(features), 1, float('inf'))
    rmse = torch.sqrt(loss(torch.log(clipped_preds),
                           torch.log(labels)))
    return rmse.item()

class ArrayDataset(Dataset):
    def __init__(self, features, labels=None):
        self.features = features
        self.labels = labels
        
    def __len__(self):
        return len(self.features)
    def __getitem__(self, idx):
        if self.labels is not None:
            self.labels=self.labels
            return self.features[idx], self.labels[idx]
        else:
            return self.features[idx]

    
def train(net, train_iter,
          num_epochs, learning_rate, weight_decay,test_labels=None,temperature=50):
    train_ls, test_ls = [], []
    # 这里使用的是Adam优化算法
    optimizer = torch.optim.Adam(net.parameters(),
                                 lr = learning_rate,
                                 weight_decay = weight_decay)
    best_rmse=torch.tensor(float('inf'))
    diff=torch.tensor(1e-4)

    flag=temperature
    start=time.time()
    for epoch in range(num_epochs):
        if device=='cuda':
            torch.cuda.synchronize()
        begin_time=time.time()
        epoch_ls=[]
        for X, y in train_iter:
            X,y=X.to(device),y.to(device)
            optimizer.zero_grad()
            l = loss(net(X), y)
            l.backward()
            optimizer.step()
            with torch.no_grad():
                epoch_ls.append(log_rmse(net, X, y))
        if device=='cuda':
            torch.cuda.synchronize()
        end_time=time.time()

        #计算rmse，保存最佳模型
        rmse=torch.mean(torch.tensor(epoch_ls)).item()
        if rmse<best_rmse and rmse-best_rmse>diff:
            best_rmse=rmse
            torch.save(net.state_dict(),'best_model.pth')
            flag=temperature
        else:
            flag-=1
        train_ls.append(rmse)

        # 打印训练进度
        if epoch%1000==0:
            print(f'epoch {epoch}, train loss {train_ls[-1]:.4f}, time: {end_time-begin_time:.2f}s')
            begin_time=end_time
        if test_labels is not None:
            with torch.no_grad():
                test_ls.append(log_rmse(net, test_features, test_labels))
    
    # 保存最新模型
    torch.save(net.state_dict(),'last_model.pth')
    end=time.time()
    print(f'train time: {end-start:.2f}')
    return train_ls, test_ls

if __name__=='__main__':
    device='cuda' if torch.cuda.is_available() else 'cpu'
    device='cuda'
    print(f'train model on {device}')
    dir_path=Path(__file__).resolve().parent
    pre_processor=PreProcessor(os.path.join(dir_path,'data/kaggle_house_pred_train.csv'),os.path.join(dir_path,'data/kaggle_house_pred_test.csv'))
    train_features,test_features,train_labels=pre_processor()
    train_features=train_features
    test_features=test_features
    train_dataset=ArrayDataset(train_features,train_labels)
    test_dataset=ArrayDataset(test_features)

    train_iter = DataLoader(train_dataset, batch_size=128, num_workers=0, shuffle=True)
    test_iter = DataLoader(test_dataset, batch_size=128, shuffle=False)

    loss=nn.MSELoss()
    in_features=train_features.shape[1]
    net=get_net(in_features)
    net=net.to(device)
    train_ls,test_ls=train(net,train_iter,num_epochs=20000,learning_rate=1e-3,weight_decay=2)
    fig,ax=plt.subplots(figsize=(5,2.7),layout='constrained')
    ax.plot(range(1, len(train_ls) + 1), train_ls,label='train loss')
    ax.plot(range(1, len(test_ls) + 1), test_ls,label='test loss')
    ax.legend()
    plt.show()