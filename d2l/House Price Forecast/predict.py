from train import get_net
from pre_process import PreProcessor
import os
from pathlib import Path

import torch

dir_path=Path(__file__).resolve().parent
device='cuda' if torch.cuda.is_available() else 'cpu'
pre_processor=PreProcessor(os.path.join(dir_path,'data/kaggle_house_pred_train.csv'),os.path.join(dir_path,'data/kaggle_house_pred_test.csv'))
train_features,test_features,train_labels=pre_processor()
in_features=test_features.shape[1]
net=get_net(in_features)
net=net.to(device)
print(net)
model_path=os.path.join(dir_path.parent.parent,'last_model.pth')
load_model=torch.load(model_path)
for name,param in load_model.items():
    print(name,param.shape)
net.load_state_dict(load_model)
net.eval()
result=[]
for test_feature in test_features:
    test_feature=test_feature.to(device)
    test_feature=test_feature.unsqueeze(0)
    with torch.no_grad():
        output=net(test_feature)
        result.append(output.item())

with open(os.path.join(dir_path,'data/kaggle_house_pred.csv'),'w') as f:
    f.write('ID,SalePrice\n')
    for i in range(len(result)):
        f.write(f'{i+1461},{result[i]}\n')
        
