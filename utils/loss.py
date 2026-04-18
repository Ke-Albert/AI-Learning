import torch
from torch import nn
from torch.nn import functional as F
from kornia.losses import dice_loss,focal_loss

def one_hot_encodding(targets,num_classes,ignore_index):
    # 将目标转换为one-hot编码
    if ignore_index is not None:
        mask=targets!=ignore_index
        targets=targets.where(mask,0)
        # [N,*]->[N,*,C]
        one_hot_targets = F.one_hot(targets, num_classes=num_classes).float()
        one_hot_targets=one_hot_targets.where(mask,ignore_index)
    else:
        one_hot_targets=F.one_hot(targets,num_classes=num_classes).float()
    return one_hot_targets.permute(0,3,1,2).contiguous()

class DiceLoss(nn.Module):
    """
    计算Dice损失
    predict: A tensor with shape [N,C,*], raw logits
    targets: A tensor with shape [N,*], ground truth labels
    weight: A tensor with shape [C], class weights
    ignore_index: An integer, index of the class to ignore or mask to ignore
    
    Reference: 
        https://github.com/hubutui/DiceLoss-PyTorch/blob/master/loss.py
        https://www.bilibili.com/video/BV1rq4y1w7xM?spm_id_from=333.788.videopod.episodes&vd_source=276daa9207bfc3366d4730ad90187a50&p=3
    """
    def __init__(self, weight=None,ignore_index=None,**kwargs):
        super().__init__()
        self.smooth = 1e-8
        self.weight=weight
        self.ignore_index=ignore_index
        self.reduction='mean'


    def _binary_dice_loss(self,predict,target):
        # d=[]
        # batch_size=predict.shape[0]
        # #每一个batch的mask掩膜不一样，需要对每个batch分开进行计算,再取平均值
        # for i in range(batch_size):
        #     x_i=predict[i].reshape(-1)
        #     t_i=target[i].reshape(-1)
        #     if self.ignore_index is not None:
        #         mask=t_i!=self.ignore_index
        #         x_i=x_i[mask]
        #         t_i=t_i[mask]
        #     inter=torch.dot(x_i,t_i)
        #     sets_sum = torch.sum(x_i) + torch.sum(t_i)
        #     if sets_sum==0:
        #         sets_sum=2*inter
        #     d.append(1-(2. * inter + self.smooth) / (sets_sum + self.smooth))
        # d=torch.tensor(d).mean()

        if self.ignore_index is not None:
            mask=target!=self.ignore_index
            target=target*mask
            predict=predict*mask
        
        inter=torch.sum(predict*target,dim=(1,2))
        sets_sum=torch.sum(predict+target,dim=(1,2))

        dice_score=(2.0*inter)/(sets_sum+self.smooth)
        dice_loss=1-dice_score

        dice_loss=dice_loss.mean()
        return dice_loss
    
    
    def forward(self, predict, targets):
        # 将输入和目标展平
        num_classes=predict.shape[1]
        targets = one_hot_encodding(targets, num_classes=num_classes,ignore_index=self.ignore_index)
        # targets: [N,C,*]
        total_loss=0
        predict=F.softmax(predict,dim=1)

        for channel in range(num_classes):
            # 忽略ignore_index通道的损失计算
            if channel!=self.ignore_index:
                dice_loss=self._binary_dice_loss(predict[:,channel,...],targets[:,channel,...])
                if self.weight:
                    assert self.weight.shape[0]==num_classes, "权重长度必须与类别数相同"
                    dice_loss*=self.weight[channel]
                total_loss+=dice_loss
                
        total_loss/=num_classes 
        # 返回Dice损失
        return total_loss

class FocalLoss(nn.Module):
    """
    predicts: A tensor with shape [N,C,*], raw logits
    targets: A tensor with shape [N,*], ground truth labels
    """
    def __init__(self, alpha=0.25, gamma=2, reduction='mean',weight=None,ignore_index=None):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        self.ignore_index=ignore_index
        self.weight=weight

    def forward(self, predicts, targets):
        num_classes=predicts.shape[1]
        targets = one_hot_encodding(targets, num_classes=num_classes,ignore_index=self.ignore_index)

        #valid_mask
        if self.ignore_index is not None:
            valid_mask=targets!=self.ignore_index
        else:
            valid_mask=torch.ones_like(targets).bool()

        targets=targets*valid_mask

        log_softmax: torch.Tensor=F.log_softmax(predicts,dim=1)
        loss_temp:torch.Tensor=-torch.pow(1.0-log_softmax.exp(),self.gamma)*log_softmax*targets

        broadcast_dims=[-1]+[1]*len(predicts.shape[2:])
        if self.alpha is not None:
            alpha_fac=torch.tensor(
                [1-self.alpha]+[self.alpha]*(num_classes-1)
            )

            alpha_fac=alpha_fac.view(broadcast_dims)
            loss_temp=alpha_fac*loss_temp
        
        if self.weight is not None:
            weight=self.weight.view(broadcast_dims)
            loss_temp=weight*loss_temp
        
        if self.reduction=='none':
            loss=loss_temp
        
        elif self.reduction=='sum':
            loss=torch.sum(loss_temp)
        elif self.reduction=='mean':
            loss=torch.mean(loss_temp)
        else:
            raise ValueError(f"Invalid reduction mode: {self.reduction}")
        
        return loss



if __name__ == '__main__':
    loss=DiceLoss()
    predict=torch.tensor([[[0.1,0.2],[0.4,0.5]],[[0.4,0.3],[0.5,0.7]]]) 
    predict=predict.unsqueeze(0)
    target=torch.tensor([[0,1],[0,1]]).long().unsqueeze(0)
    loss_v1=loss(predict,target)
    loss_v3=dice_loss(predict,target,average='macro')
    print('loss_v1:',loss_v1.item(),'loss_v3:',loss_v3.item())

    
    loss_v2=FocalLoss()(predict,target)
    loss_v4=focal_loss(predict,target,alpha=0.25,reduction='mean')
    print('loss_v2:',loss_v2.item(),'loss_v4:',loss_v4.item())
