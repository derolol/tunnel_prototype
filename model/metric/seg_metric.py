import einops

import torch
from torch.nn import Module
import torch.nn.functional as F

# https://torchmetrics.readthedocs.io/en/stable/classification/accuracy.html
from torchmetrics.classification import MulticlassFBetaScore, MulticlassAccuracy, MulticlassPrecision, MulticlassRecall
from torchmetrics.detection import iou

class SegFBetaScoreMetric(Module):
    
    def __init__(self, num_classes, beta=1.) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.metric = MulticlassFBetaScore(num_classes=num_classes, beta=beta, average=None)

    def forward(self, preds, target):
        assert len(preds.shape) == 4 # [B, C, H, W]
        assert len(target.shape) == 3 # [B, H, W]

        preds = einops.rearrange(preds, 'b c h w -> (b h w) c')
        target = einops.rearrange(target, 'b h w -> (b h w)')
        
        return self.metric.to(preds)(preds, target)

class SegPrecisionMetric(Module):
    
    def __init__(self, num_classes) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.metric = MulticlassPrecision(num_classes=num_classes, average=None)

    def forward(self, preds, target):
        assert len(preds.shape) == 4 # [B, C, H, W]
        assert len(target.shape) == 3 # [B, H, W]

        preds = einops.rearrange(preds, 'b c h w -> (b h w) c')
        target = einops.rearrange(target, 'b h w -> (b h w)')
        
        return self.metric.to(preds)(preds, target)

class SegRecallMetric(Module):
    
    def __init__(self, num_classes) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.metric = MulticlassRecall(num_classes=num_classes, average=None)

    def forward(self, preds, target):
        assert len(preds.shape) == 4 # [B, C, H, W]
        assert len(target.shape) == 3 # [B, H, W]

        preds = einops.rearrange(preds, 'b c h w -> (b h w) c')
        target = einops.rearrange(target, 'b h w -> (b h w)')
        
        return self.metric.to(preds)(preds, target)

class SegAccuracyMetric(Module):
    
    def __init__(self, num_classes) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.metric = MulticlassAccuracy(num_classes=num_classes, average=None)

    def forward(self, preds, target):
        assert len(preds.shape) == 4 # [B, C, H, W]
        assert len(target.shape) == 3 # [B, H, W]
        preds = einops.rearrange(preds, 'b c h w -> (b h w) c')
        target = einops.rearrange(target, 'b h w -> (b h w)')
        
        return self.metric.to(preds)(preds, target)

class SegIoUMetric(Module):
    
    def __init__(self, num_classes) -> None:
        super().__init__()
        self.num_classes = num_classes
        # self.metric = MulticlassAccuracy(num_classes=num_classes, average=None)

    def forward(self, preds, target):
        assert len(preds.shape) == 4 # [B, C, H, W]
        assert len(target.shape) == 3 # [B, H, W]
        
        preds = preds.argmax(dim=1)
        
        # 计算hist矩阵
        hist = self.get_hist(preds.flatten(), target.flatten(), self.num_classes)  
        
        # 计算各类别的IoU
        insert = hist.diag()
        union = hist.sum(dim=1) + hist.sum(dim=0) - hist.diag()
        union = union.maximum(torch.ones_like(hist.diag()))
        ious = insert / union

        return ious

    def get_hist(self, pred, label, n):
        '''
        获取混淆矩阵
        label 标签 一维数组 HxW
        pred 预测结果 一维数组 HxW
        '''
        k = (label >= 0) & (label < n)
        # 对角线上的为分类正确的像素点
        return torch.bincount(n * label[k] + pred[k], minlength=n ** 2).reshape((n, n))