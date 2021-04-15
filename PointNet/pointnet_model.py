import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import scipy.spatial.distance
import math
import random

class Tnet(nn.Module):
    def __init__(self,k=3):
        super().__init__()
        self.k = k
        self.mlp1 = nn.Conv1d(k,64,1)
        self.mlp2 = nn.Conv1d(64,128,1)
        self.mlp3 = nn.Conv1d(128,1024,1)

        self.fc1 = nn.Linear(1024,512)
        self.fc2 = nn.Linear(512,128)
        self.fc3 = nn.Linear(128,k*k)

        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)
        self.bn4 = nn.BatchNorm1d(512)
        self.bn5 = nn.BatchNorm1d(128)

    def forward(self,input):
        bs = input.size(0) #input.size = batch_size*n*3 (but it is transposed when it enters)
        xb = F.relu(self.bn1(self.mlp1(input))) #the input is already transposed(1,2)
        xb = F.relu(self.bn2(self.mlp2(xb)))
        xb = F.relu(self.bn3(self.mlp3(xb)))
        pool = nn.MaxPool1d(xb.size(-1))(xb) #-1 indicates final dimension, becomes 1 dimension bs*dim*n=>bs*dim*1
        flat = nn.Flatten(1)(pool) #start-dim = 1, for fully-connected layers, you need to flatten the model bs*dim
        
        xb = F.relu(self.bn4(self.fc1(flat)))
        xb = F.relu(self.bn5(self.fc2(xb)))
        
        init = torch.eye(self.k).repeat(bs,1,1)
        matrix = self.fc3(xb).view(-1,self.k,self.k) + init.cuda()
        return matrix
        
        
class feature_transform(nn.Module):
    def __init__(self):
        super().__init__()
        self.input_transform = Tnet(k=3)
        self.feature_transform = Tnet(k=64)
        
        self.mlp1 = nn.Conv1d(3,64,1)
        self.mlp2 = nn.Conv1d(64,128,1)
        self.mlp3 = nn.Conv1d(128,1024,1)
        
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)
        
    def forward(self,input):
        matrix3x3 = self.input_transform(input) #
        xb = torch.bmm(torch.transpose(input,1,2),matrix3x3).transpose(1,2) #becomes batch_size*channel*n_pts where channel is 3
        xb = F.relu(self.bn1(self.mlp1(xb))) #batch_size*64*n_pts
        matrix64x64 = self.feature_transform(xb) #batch_size*64*64
        xb = torch.bmm(torch.transpose(xb,1,2),matrix64x64).transpose(1,2) #since it is transposed, n_pts*64 * 64*64 => batch_size*n_pts*64
        xb = F.relu(self.bn2(self.mlp2(xb)))
        xb = F.relu(self.bn3(self.mlp3(xb))) #xb shape => batch_size*1024*n_pts
        pool = nn.MaxPool1d(xb.size(-1))(xb) #Global max pool the final dimension which is n_pts, hence the result is 1024 features
        flat = nn.Flatten(1)(pool) #pool -> batch_size*1024*1 ->flatten(start_dim=1) -> batch_size*1024
        
        return flat,matrix3x3, matrix64x64
        
class Pointnet_class(nn.Module):
    def __init__(self,classes=10):
        super().__init__()
        self.transform = feature_transform()
        self.fc1 = nn.Linear(1024,512)
        self.fc2 = nn.Linear(512,256)
        self.fc3 = nn.Linear(256, classes)
        
        self.bn1 = nn.BatchNorm1d(512)
        self.bn2 = nn.BatchNorm1d(256)
        
        self.dropout = nn.Dropout(p=0.3)
        self.logsoftmax = nn.LogSoftmax(dim=1)
        
        
    def forward(self,input):
        xb, matrix3x3, matrix64x64 = self.transform(input)
        #print("xb size: ",xb.size())
        xb = F.relu(self.bn1(self.fc1(xb)))
        xb = F.relu(self.bn2(self.dropout(self.fc2(xb))))
        xb = self.fc3(xb)
        output = self.logsoftmax(xb)
        return output, matrix3x3, matrix64x64
    
