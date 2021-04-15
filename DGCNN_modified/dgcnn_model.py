import torch
import numpy as np
import math
import random
import utils
import random
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F

def one_hot(labels, num_classes):
    """Embedding labels to one-hot form.

    Args:
      labels: (LongTensor) class labels, sized [N,].
      num_classes: (int) number of classes.

    Returns:
      (tensor) encoded labels, sized [N, #classes].
    """
    y = torch.eye(num_classes)
    return y[labels] 

def knn(x, k):
    inner = -2*torch.matmul(x.transpose(2, 1), x)
    xx = torch.sum(x**2, dim=1, keepdim=True)
    pairwise_distance = -xx - inner - xx.transpose(2, 1)
 
    idx = pairwise_distance.topk(k=k, dim=-1)[1]   # (batch_size, num_points, k)
    return idx


def knn_dilate(x, k):
    batch_size = x.size(0)
    device = torch.device('cuda')
    k2 = 2*k
    inner = -2*torch.matmul(x.transpose(2, 1), x)
    xx = torch.sum(x**2, dim=1, keepdim=True)
    pairwise_distance = -xx - inner - xx.transpose(2, 1)
 
    idx = pairwise_distance.topk(k=k2, dim=-1)[1]   # (batch_size, num_points, k)
    #idx = idx.squeeze(0) #(num_points, k)
    idx = idx.cpu().detach().numpy()
    idx_i = np.random.choice(idx.shape[2],k,replace=False)
    #idx_i = np.sort(idx_i)
    idx = idx[:, :, idx_i]
    #print(idx.shape)
    idx = torch.tensor(idx).to(device)
    return idx

def get_graph_feature(x, k, idx=None, dim9=False, dilation=True):
    batch_size = x.size(0)
    num_points = x.size(2)
    x = x.view(batch_size, -1, num_points)
    if idx is None:
        if dim9 == False:
            if dilation == True:
                idx = knn_dilate(x, k=k)
            else:
                idx = knn(x, k=k)   # (batch_size, num_points, k)
            #print(idx)
        else:
            idx = knn(x[:, 6:], k=k)
    device = torch.device('cuda')

    idx_base = torch.arange(0, batch_size, device=device).view(-1, 1, 1)*num_points
    #idx_base = torch.arange(0, batch_size).view(-1, 1, 1)*num_points

    idx = idx + idx_base

    idx = idx.contiguous().view(-1)
 
    _, num_dims, _ = x.size()

    x = x.transpose(2, 1).contiguous()   # (batch_size, num_points, num_dims)  -> (batch_size*num_points, num_dims) #   batch_size * num_points * k + range(0, batch_size*num_points)
    feature = x.view(batch_size*num_points, -1)[idx, :]
    feature = feature.view(batch_size, num_points, k, num_dims) 
    x = x.view(batch_size, num_points, 1, num_dims).repeat(1, 1, k, 1)
    
    feature = torch.cat((feature-x, x), dim=3).permute(0, 3, 1, 2).contiguous()
  
    return feature      # (batch_size, 2*num_dims, num_points, k)

class SE_Block(nn.Module):
    "credits: https://github.com/moskomule/senet.pytorch/blob/master/senet/se_module.py#L4"
    def __init__(self, c, r=16):
        super().__init__()
        self.squeeze = nn.AdaptiveAvgPool1d(1)
        self.excitation = nn.Sequential(
            nn.Linear(c, c // r, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(c // r, c, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        bs, c, _, = x.shape
        y = self.squeeze(x).view(bs, c)
        y = self.excitation(y).view(bs, c, 1)
        return x * y.expand_as(x)
    
class Transform_Net(nn.Module):
    def __init__(self,k):#, args):
        super(Transform_Net, self).__init__()
        #self.args = args
        self.k = k

        self.bn1 = nn.GroupNorm(32,64)
        self.bn1_2 = nn.GroupNorm(32,64)
        self.bn2 = nn.GroupNorm(32,128)
        self.bn3 = nn.GroupNorm(32,1024)
        
        self.conv1 = nn.Sequential(nn.Conv2d(self.k*2, 64, kernel_size=1, bias=False),
                                   self.bn1,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv1_2 = nn.Sequential(nn.Conv2d(64, 64, kernel_size=1, bias=False),
                                    self.bn1_2,
                                    nn.LeakyReLU(negative_slope=0.2))
        self.conv2 = nn.Sequential(nn.Conv2d(64, 128, kernel_size=1, bias=False),
                                   self.bn2,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv3 = nn.Sequential(nn.Conv1d(128, 1024, kernel_size=1, bias=False),
                                   self.bn3,
                                   nn.LeakyReLU(negative_slope=0.2))

        self.linear1 = nn.Linear(1024, 512, bias=False)
        self.bn3 = nn.GroupNorm(32,512)
        self.linear2 = nn.Linear(512, 256, bias=False)
        self.bn4 = nn.GroupNorm(32,256)

        self.transform = nn.Linear(256, k*k)#3*3)
        init.constant_(self.transform.weight, 0)
        init.eye_(self.transform.bias.view(k, k))#3, 3))

    def forward(self, x):
        batch_size = x.size(0)
        
        x = self.conv1(x)                       # (batch_size, 3*2, num_points, k) -> (batch_size, 64, num_points, k)
        x = self.conv1_2(x)
        x = self.conv2(x)                       # (batch_size, 64, num_points, k) -> (batch_size, 128, num_points, k)
        x = x.max(dim=-1, keepdim=False)[0]     # (batch_size, 128, num_points, k) -> (batch_size, 128, num_points)

        x = self.conv3(x)                       # (batch_size, 128, num_points) -> (batch_size, 1024, num_points)
        x = x.max(dim=-1, keepdim=False)[0]     # (batch_size, 1024, num_points) -> (batch_size, 1024)

        x = F.leaky_relu(self.bn3(self.linear1(x)), negative_slope=0.2)     # (batch_size, 1024) -> (batch_size, 512)
        x = F.leaky_relu(self.bn4(self.linear2(x)), negative_slope=0.2)     # (batch_size, 512) -> (batch_size, 256)
        
        #initialize as identity
        init = torch.eye(self.k, requires_grad=True).repeat(batch_size,1,1)
        if x.is_cuda:
            init=init.cuda()
        x = self.transform(x)                   # (batch_size, 256) -> (batch_size, 3*3)
        x = x.view(batch_size, self.k,self.k) + init            # (batch_size, 3*3) -> (batch_size, 3, 3)

        return x

class EdgeConv2(nn.Module):
    def __init__(self,k, in_dims, layer=2, norm = 'batch'):
        super(EdgeConv2, self).__init__()
        self.k = k
        self.in_dims = in_dims
        self.x = 64
        self.norm = norm
        self.layer = layer
        if self.norm == 'batch':
            self.bn1 = nn.BatchNorm2d(self.x)
            self.bn2 = nn.BatchNorm2d(self.x)
            self.bn1_2 = nn.BatchNorm2d(self.x)
            self.bn2_2 = nn.BatchNorm2d(self.x)
            self.bn_cat = nn.BatchNorm2d(self.x)
            self.bn_cat2 = nn.BatchNorm2d(self.x)
            self.bn_add1 = nn.BatchNorm1d(self.x)
        else:
            self.bn1 = nn.GroupNorm(32,self.x)
            self.bn2 = nn.GroupNorm(32,self.x)
            self.bn1_2 = nn.GroupNorm(32,self.x)
            self.bn2_2 = nn.GroupNorm(32,self.x)
            self.bn_cat = nn.GroupNorm(32,self.x)
            self.bn_cat2 = nn.GroupNorm(32,self.x)
            self.bn_add1 = nn.GroupNorm(32,self.x)
        
        self.conv1 = nn.Sequential(nn.Conv2d(self.in_dims, self.x, kernel_size=1, bias=False),
                                   self.bn1,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv2 = nn.Sequential(nn.Conv2d(self.x, self.x, kernel_size=1, bias=False),
                                   self.bn2,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv1_2 = nn.Sequential(nn.Conv2d(self.in_dims, self.x, kernel_size=1, bias=False),
                                   self.bn1_2,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv2_2 = nn.Sequential(nn.Conv2d(self.x, self.x, kernel_size=1, bias=False),
                                   self.bn2_2,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv_cat = nn.Sequential(nn.Conv1d(self.x*2,self.x,kernel_size=1, bias=False),
                                     self.bn_cat,
                                     nn.LeakyReLU(negative_slope=0.2))
        self.conv_cat2 = nn.Sequential(nn.Conv1d(self.x*2,self.x,kernel_size=1, bias=False),
                                     self.bn_cat2,
                                     nn.LeakyReLU(negative_slope=0.2))
        self.conv_add1 = nn.Sequential(nn.Conv1d(self.x*2,self.x,kernel_size=1,bias=False),
                                      self.bn_add1,
                                      nn.LeakyReLU(negative_slope=0.2))
        self.se1 = SE_Block(self.x)
        self.se1_2 = SE_Block(self.x)

        
    def forward(self, x):
        if self.layer == 2:
            x1 = get_graph_feature(x, k=self.k, dilation=True)      # (batch_size, 3, num_points) -> (batch_size, 3*2, num_points, k)
            x1 = self.conv1(x1)
            x1 = self.conv2(x1)
            x1_max = x1.max(dim=-1, keepdim=False)[0]    # (batch_size, 64, num_points, k) -> (batch_size, 64, num_points)
            x1_mean = x1.mean(dim=-1, keepdim=False)    # (batch_size, 64, num_points, k) -> (batch_size, 64, num_points)
            x1 = torch.cat((x1_max,x1_mean),1)
            x1 = self.conv_cat(x1)
            x1 = self.se1(x1)

            x2 = get_graph_feature(x, k=self.k, dilation=False)      # (batch_size, 3, num_points) -> (batch_size, 3*2, num_points, k)
            x2 = self.conv1_2(x2)                       # (batch_size, 3*2, num_points, k) -> (batch_size, 64, num_points, k)
            x2 = self.conv2_2(x2)                       # (batch_size, 64, num_points, k) -> (batch_size, 64, num_points, k)
            x1_2 = x2.max(dim=-1, keepdim=False)[0]    # (batch_size, 64, num_points, k) -> (batch_size, 64, num_points)
            x1_2_mean = x2.mean(dim=-1, keepdim=False)    # (batch_size, 64, num_points, k) -> (batch_size, 64, num_points)
            x1_2 = torch.cat((x1_2,x1_2_mean),1)
            x1_2 = self.conv_cat2(x1_2)##
            x1_2 = self.se1_2(x1_2)
            x1 = torch.cat((x1,x1_2),1)
            x1 = self.conv_add1(x1)
            residual = x1
        elif self.layer == 1:
            x1 = get_graph_feature(x, k=self.k, dilation=True)
            x1 = self.conv1(x1)
            x1_max = x1.max(dim=-1, keepdim=False)[0]    # (batch_size, 64, num_points, k) -> (batch_size, 64, num_points)
            x1_mean = x1.mean(dim=-1, keepdim=False)    # (batch_size, 64, num_points, k) -> (batch_size, 64, num_points)
            x1 = torch.cat((x1_max,x1_mean),1)
            x1 = self.conv_cat(x1)
            x1 = self.se1(x1)
            
            x2 = get_graph_feature(x, k=self.k, dilation=False)      # (batch_size, 3, num_points) -> (batch_size, 3*2, num_points, k)
            x2 = self.conv1_2(x2)                       # (batch_size, 3*2, num_points, k) -> (batch_size, 64, num_points, k)
            x1_2 = x2.max(dim=-1, keepdim=False)[0]    # (batch_size, 64, num_points, k) -> (batch_size, 64, num_points)
            x1_2_mean = x2.mean(dim=-1, keepdim=False)    # (batch_size, 64, num_points, k) -> (batch_size, 64, num_points)
            x1_2 = torch.cat((x1_2,x1_2_mean),1)
            x1_2 = self.conv_cat2(x1_2)##
            x1_2 = self.se1_2(x1_2)
            x1 = torch.cat((x1,x1_2),1)
            x1 = self.conv_add1(x1)
            residual = x1

        return residual
    
class DGCNN(nn.Module):
    def __init__(self, seg_num_all, norm = 'batch'):
        super(DGCNN, self).__init__()
        #self.args = args
        self.norm = norm
        self.seg_num_all = seg_num_all
        self.k = 20
        self.x = 64
        self.transform_net = Transform_Net(3)
        self.edge1 = EdgeConv2(k=20, in_dims=6, layer=2, norm='group')
        self.edge2 = EdgeConv2(k=20, in_dims=self.x*2, layer=2, norm='group')
        self.edge3 = EdgeConv2(k=20, in_dims=self.x*2, layer=1, norm='group')
        
        if self.norm == 'batch':
            self.bn1 = nn.BatchNorm1d(1024)
            self.bn2 = nn.BatchNorm1d(1024)
            self.bn3 = nn.BatchNorm1d(512)
            self.bn4 = nn.BatchNorm1d(512)
            self.bn5 = nn.BatchNorm1d(512)
        elif self.norm == 'group':
            self.bn1 = nn.GroupNorm(32, 1024)
            self.bn2 = nn.GroupNorm(32, 1024)
            self.bn3 = nn.GroupNorm(32, 512)
            self.bn4 = nn.GroupNorm(32, 512)
            self.bn5 = nn.GroupNorm(32, 512)
            
        self.conv1 = nn.Sequential(nn.Conv1d(192, 1024, kernel_size=1, bias=False),
                                   self.bn1,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv2 = nn.Sequential(nn.Conv1d(2048, 1024, kernel_size=1, bias=False),
                                  self.bn2,
                                  nn.LeakyReLU(negative_slope=0.2))
        self.conv3 = nn.Sequential(nn.Conv1d(1024+64*3, 512, kernel_size=1, bias=False),
                                  self.bn3,
                                  nn.LeakyReLU(negative_slope=0.2))
        self.conv4 = nn.Sequential(nn.Conv1d(512, 512, kernel_size=1, bias=False),
                                  self.bn4,
                                  nn.LeakyReLU(negative_slope=0.2))
        self.conv5 = nn.Sequential(nn.Conv1d(512, 512, kernel_size=1, bias=False),
                                  self.bn5,
                                  nn.LeakyReLU(negative_slope=0.2))
        self.conv6 = nn.Conv1d(512, self.seg_num_all, kernel_size=1, bias=False)
        self.se3 = SE_Block(512)
        self.se4 = SE_Block(512)
        self.se5 = SE_Block(512)
        
        self.dp1 = nn.Dropout(p=0.5)
        self.dp2 = nn.Dropout(p=0.5)
        self.dp3 = nn.Dropout(p=0.5)
                                
        
    def forward(self, x, l):
        batch_size = x.size(0)
        num_points = x.size(2)

        x0 = get_graph_feature(x, k=self.k, dilation=True)     # (batch_size, 3, num_points) -> (batch_size, 3*2, num_points, k)
        t = self.transform_net(x0)              # (batch_size, 3, 3)
        matrix3x3 = t
        x = x.transpose(2, 1)                   # (batch_size, 3, num_points) -> (batch_size, num_points, 3)
        x = torch.bmm(x, t)                     # (batch_size, num_points, 3) * (batch_size, 3, 3) -> (batch_size, num_points, 3)
        x = x.transpose(2, 1)                   # (batch_size, num_points, 3) -> (batch_size, 3, num_points)
        
        x1 = self.edge1(x)
        x2 = self.edge2(x1)
        x2 += x1
        
        x3 = self.edge3(x2)
        x3 += x2
        
        x = torch.cat((x1, x2, x3), dim=1)      # (batch_size, 64*3, num_points)
        x = self.conv1(x)                       # (batch_size, 64*3, num_points) -> (batch_size, emb_dims, num_points)
        x_max = x.max(dim=-1, keepdim=True)[0]      # (batch_size, emb_dims, num_points) -> (batch_size, emb_dims, 1)
        x_mean = x.mean(dim=-1, keepdim=True)
        x = torch.cat((x_max,x_mean),1)
        x = self.conv2(x)
        
        x = x.repeat(1, 1, num_points)          # (batch_size, 1088, num_points)

        x = torch.cat((x, x1, x2, x3), dim=1)   # (batch_size, 1088+64*3, num_points)
        
        x = self.conv3(x)                       # (batch_size, 1088+64*3, num_points) -> (batch_size, 256, num_points)
        x = self.se3(x)
        residual = x
        x = self.dp1(x)
        
        x = self.conv4(x)                       # (batch_size, 256, num_points) -> (batch_size, 256, num_points)
        x = self.se4(x)
        x+=residual
        residual = x
        x = self.dp2(x)
        
        x = self.conv5(x)                      # (batch_size, 256, num_points) -> (batch_size, 128, num_points)
        self.se5(x)
        x+=residual
        x = self.dp3(x)
        x = self.conv6(x)                      # (batch_size, 256, num_points) -> (batch_size, seg_num_all, num_points)
        
        return x