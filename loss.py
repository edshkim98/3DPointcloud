import torch
import numpy as np
import os
import open3d as o3d

def diceLoss(prediction_g, label_g, num_class, epsilon=1):
    ls = []
    diceRatio_g = 0
    label_g = label_g.reshape(batch_size*2048, -1)##########################
    prediction_g = prediction_g.reshape(batch_size*2048,-1)##############################
    for i in range(num_class):
        pred = prediction_g
        label = label_g
        pred=torch.nn.functional.softmax(pred, dim=1)[:, i]
        pred = pred.reshape(-1,1) #bs*pts*1

        
        diceLabel_g = label.sum(dim=0)
        diceLabel_g = diceLabel_g[i]
        
        dicePrediction_g = pred.sum(dim=0)
        
        diceCorrect_g = (pred * label)[:,i]
        #print(diceCorrect_g)
        diceCorrect_g = diceCorrect_g.sum()
        #print(diceCorrect_g)
        #print(diceLabel_g,dicePrediction_g,diceCorrect_g)
        
        
        diceRatio_g += (2 * diceCorrect_g + epsilon) \
        / (dicePrediction_g + diceLabel_g + epsilon)
        
    loss = 1-(1/num_class)*diceRatio_g
    #print(loss)
    
    return loss

def dgcnn_loss(pred,labels, smoothing = True):
    
    #labels = labels.contiguous().view(-1)
    
    if smoothing:
        labels = labels.view(-1,1).squeeze()
        labels = labels.contiguous().view(-1)
        pred = pred.permute(0, 2, 1).contiguous()
        pred = pred.view(-1, 3)#####################
        eps = 0.2
        n_class = pred.size(1)
        
#        print(pred.shape)
#        one_hot = torch.zeros_like(pred)
#        print(one_hot.shape)
#        print(labels.view(-1,1).shape)
#        one_hot = one_hot.scatter(1,labels.view(-1,1),1)
        one_hot_tensor = torch.zeros_like(pred).scatter(1, labels.view(-1,1), 1)
        one_hot_tensor = one_hot_tensor * (1-eps) + (1-one_hot_tensor) * eps / (n_class - 1)
        log_prb = F.log_softmax(pred, dim=1)
        
        loss = -(one_hot_tensor * log_prb).sum(dim=-1).mean()
    else:
        criterion = torch.nn.CrossEntropyLoss()
    
        loss1 = criterion(pred, labels)
        loss = loss1 
    
    labels_onehot = one_hot(labels,3).to(device)#####################
    dice_loss = diceLoss(pred,labels_onehot,3)######################
    loss = loss + dice_loss
    return loss

#we want to make the matrix orthogonal => q*q^T = I
#Hence, if the matrix is orthogonal -> I - q*q^T = I-I = 0
def pointnet_loss(outputs,labels,matrix3x3, matrix64x64,alpha= 0.0001):
    criterion = torch.nn.NLLLoss()
    bs = outputs.size(0)
    im3x3 = torch.eye(3,requires_grad=True).repeat(bs,1,1).cuda()
    im64x64 = torch.eye(64,requires_grad=True).repeat(bs,1,1).cuda()
    
    diff3x3 = im3x3 - torch.bmm(torch.transpose(matrix3x3,1,2),matrix3x3)
    diff64x64 = im64x64 - torch.bmm(torch.transpose(matrix64x64,1,2),matrix64x64)
    loss = criterion(outputs,labels)
    return loss + alpha*(torch.norm(diff3x3)+torch.norm(diff64x64)) / bs