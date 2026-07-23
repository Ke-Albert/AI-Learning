import torch
from torch import nn

bs,ts,embed_dim=2,3,6
num_groups=2
inputx=torch.randn(bs,ts,embed_dim) #B*L*C

# 1. test batchnorm and its api
#NLP:[B,L,C]->[C]
#CV:[B,C,H,W]->[C]
batch_norm_op=nn.BatchNorm1d(embed_dim)
bn_y=batch_norm_op(inputx.transpose(1, 2)).transpose(1, 2)

# 手写batch_norm
def batch_norm(x):
    eps=1e-5
    gamma=1
    beta=0
    mean=x.mean(dim=(0,1),keepdim=True)
    std=x.std(dim=(0,1),keepdim=True,unbiased=False) #有偏估计
    return ((x-mean)/(std+eps))*gamma+beta
verify_y=batch_norm(inputx)
print(torch.allclose(verify_y,bn_y,rtol=1e-3))

#2. test layer norm and its api
#NLP:[B,L,C]->[B,L]
#CV:[B,C,H,W]->[B,H,W]
layer_norm_op=nn.LayerNorm(embed_dim,elementwise_affine=False)
ln_y=layer_norm_op(inputx)

def layer_norm(x):
    eps=1e-5
    mean=x.mean(dim=-1,keepdim=True)
    std=x.std(dim=-1,keepdim=True,unbiased=False)
    return (x-mean)/(std+eps)

verify_lny=layer_norm(inputx)
print(torch.allclose(verify_lny,ln_y,rtol=1e-3))

#3.test instance norm and its api
#NLP:[B,L,C]->[B,C]
#CV:[B,C,H,W]->[B,C]
ins_norm_op=nn.InstanceNorm1d(embed_dim)
ins_y=ins_norm_op(inputx.transpose(1, 2)).transpose(1, 2)

def ins_norm(x):
    eps=1e-5
    mean=x.mean(dim=1,keepdim=True)
    std=x.std(dim=1,keepdim=True,unbiased=False)
    return (x-mean)/(std+eps)
verify_ins=ins_norm(inputx)
print(torch.allclose(verify_ins,ins_y,rtol=1e-3))

#4.test groupnorm and its api
#NLP:[B,G,L,C//G]->[B,G]
#CV:[B,G,C//G,H,W]->[B,G]
group_norm_op=nn.GroupNorm(num_groups,embed_dim,affine=False)
g_y=group_norm_op(inputx.transpose(1, 2)).transpose(1, 2)

def group_norm(x):
    eps=1e-5
    x_split=x.split(embed_dim//num_groups,dim=-1)
    result=[]
    for group in x_split:
        print(group.shape)
        mean=group.mean(dim=(1,2),keepdim=True)
        std=group.std(dim=(1,2),keepdim=True,unbiased=False)
        result.append((group-mean)/(std+eps))
    result=torch.cat(result,dim=-1)
    return result
verify_group=group_norm(inputx)
print(torch.allclose(verify_group,g_y,rtol=1e-3))

# 5.test weightnorm and its api
linear=nn.Linear(embed_dim,3,bias=False)
wn_linear=nn.utils.weight_norm(linear)

wn_linear_output=wn_linear(inputx)
weight_direction=linear.weight/linear.weight.norm(dim=1,keepdim=True)
weight_magnitude=wn_linear.weight_g
print(weight_direction.shape,weight_magnitude.shape)
verify_wn_linear=inputx@(weight_direction.transpose(-1,-2))*weight_magnitude.transpose(-1,-2)
print(torch.allclose(verify_wn_linear,wn_linear_output,rtol=1e-3))