# custom metric
import torch 
import torch.nn as nn
import torch.nn.functional as F

def iou_score(output, target):
    smooth = 1e-5

    if torch.is_tensor(output):
        output = torch.sigmoid(output).data.cpu().numpy()
    if torch.is_tensor(target):
        target = target.data.cpu().numpy()
    output_ = output > 0.5
    target_ = target > 0.5
    intersection = (output_ & target_).sum()
    union = (output_ | target_).sum()

    return (intersection + smooth) / (union + smooth)


def dice_coef(output, target):
    smooth = 1e-5

    output = torch.sigmoid(output).view(-1).data.cpu().numpy()
    target = target.view(-1).data.cpu().numpy()
    intersection = (output * target).sum()

    return (2. * intersection + smooth) / (output.sum() + target.sum() + smooth)
    
def dice_coef_loss(y_true, y_pred):
    return 1.0 - dice_coef(y_true, y_pred)
    #return 1.0 - dice_coef(y_true, y_pred)
     
# def jaccard_coef(y_true, y_pred, smooth=0.0):
#     '''Average jaccard coefficient per batch.'''
#     axes = (1,2,3)
#     intersection = K.sum(y_true * y_pred, axis=axes)
#     union = K.sum(y_true + y_pred, axis=axes) - intersection
#     return K.mean( (intersection + smooth) / (union + smooth), axis=0)

def accuracy(pred, label):
    pred = torch.sigmoid(pred)
    temp = torch.zeros(pred.shape[0],pred.shape[1],pred.shape[2],pred.shape[3]).cuda()
    temp[pred>=0.5] = 255
    temp[pred<0.5] = 0
    corrects = (temp == label).float()
    acc = corrects.sum() / corrects.numel()
    return acc.item()




def dice_loss(pred, target, smooth=1.):
    pred = pred.contiguous()
    target = target.contiguous()

    intersection = (pred * target).sum(dim=2).sum(dim=2)

    loss = (1 - ((2. * intersection + smooth) /
            (pred.sum(dim=2).sum(dim=2) + target.sum(dim=2).sum(dim=2) + smooth)))

    return loss.mean()

def calc_loss(pred, target, metrics, bce_weight=0.5):
    bce = F.binary_cross_entropy_with_logits(pred, target)
    pred = F.sigmoid(pred)
    dice = dice_loss(pred, target)

    loss = bce * bce_weight + dice * (1 - bce_weight)

    metrics['bce'] += bce.data.cpu().numpy() * target.size(0)
    metrics['dice'] += dice.data.cpu().numpy() * target.size(0)
    metrics['loss'] += loss.data.cpu().numpy() * target.size(0)

    return loss
