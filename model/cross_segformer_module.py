# Ref:
# [1] Code: https://github.com/miracleyoo/pytorch-lightning-template
# [2] Version2: new branch fea: feature_backgound x feature_defect_edge

import numpy as np
import einops
from omegaconf import OmegaConf

import torch
from torch.nn import Conv2d
from torch.nn import functional as F
from torch.nn import MSELoss
from torch.optim import Adam
from torch.optim.lr_scheduler import OneCycleLR
from torchvision import utils as vutils

from pytorch_lightning import LightningModule

from model.loss.focal_loss import FocalLoss
from model.metric.seg_metric import SegAccuracyMetric, SegFBetaScoreMetric, SegPrecisionMetric, SegRecallMetric, SegIoUMetric
from util.common import instantiate_from_config, load_state_dict

class ModelModule(LightningModule):

    def __init__(self,
                 model_config,
                 resume,
                 learning_rate,
                 num_classes,
                 type_classes,
                 color_map,
                 *args, **kwargs) -> None:
        
        super().__init__(*args, **kwargs)

        self.model_config = model_config

        self.learning_rate = learning_rate
        self.num_classes =num_classes
        self.type_classes = type_classes
        self.color_map = torch.tensor(color_map)

        self.model = instantiate_from_config(OmegaConf.load(model_config))
        if resume:
            state_dict = torch.load(resume, map_location="cpu")
            load_state_dict(model=self.model,
                            state_dict=state_dict,
                            strict=True)
        
        self.configure_losses()
        self.configure_metics()
    
    def configure_losses(self):
        self.loss_fn = {
            'segment_loss': FocalLoss(),
            'edge_loss': MSELoss(),
        }
    
    def configure_metics(self):
        self.metric_fn = {
            'segment_accuracy': SegAccuracyMetric(num_classes=self.num_classes),
            'segment_f1score': SegFBetaScoreMetric(num_classes=self.num_classes),
            'segment_precision': SegPrecisionMetric(num_classes=self.num_classes),
            'segment_recall': SegRecallMetric(num_classes=self.num_classes),
            'segment_iou': SegIoUMetric(num_classes=self.num_classes),
        }
            
    def configure_optimizers(self):
        optimizer = Adam(params=self.parameters(),
                         lr=self.learning_rate)

        scheduler = OneCycleLR(optimizer=optimizer,
                               max_lr=self.learning_rate,
                               total_steps=self.trainer.max_steps)
        
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
                "frequency": 1,
                "strict": False
            },
        }

    def forward(self, img):
        return self.model(img)
    
    def log(self, name, value):
        super().log(name=name,
                    value=value,
                    on_step=True,
                    on_epoch=True,
                    prog_bar=True,
                    sync_dist=True,
                    logger=True)
    
    def get_conv2d(self, in_channels, value):
        # 用nn.Conv2d定义卷积操作
        conv_op = Conv2d(in_channels=in_channels,
                          out_channels=in_channels,
                          kernel_size=3,
                          padding=1,
                          bias=False,
                          groups=in_channels).to(self.device)
        # 定义Prewitt算子参数
        prewitt_kernel = np.array(value, dtype='float32')
        # 将Prewitt算子转换为适配卷积操作的卷积核
        prewitt_kernel = prewitt_kernel.reshape((1, 1, 3, 3))
        conv_op.weight.data = torch.from_numpy(prewitt_kernel).to(self.device).repeat(in_channels, 1, 1, 1)

        return conv_op

    def edge_conv2d(self, annotation):
        with torch.no_grad():

            B, H, W = annotation.shape
            
            conv_op1 = self.get_conv2d(in_channels=B, value=[[-1, -1, -1], [0, 0, 0], [1, 1, 1]])
            conv_op2 = self.get_conv2d(in_channels=B, value=[[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]])
            edge_detect1 = conv_op1(annotation.float().unsqueeze(dim=0)).squeeze(dim=0)
            edge_detect2 = conv_op2(annotation.float().unsqueeze(dim=0)).squeeze(dim=0)

            # 边缘标识为2, 连通区域标识为0, 背景标识为1
            edge_detect = edge_detect1.abs() + edge_detect2.abs()
            edge_detect[edge_detect > 0] = 2
            edge_detect[annotation == 0] = 1

            edge_detect = F.max_pool2d(edge_detect.unsqueeze(dim=1),
                                       kernel_size=7,
                                       stride=4,
                                       padding=3).squeeze(dim=1)

            return edge_detect

    def calculate_loss(self, out, batch, batch_idx, mode):
        images, annotations, labels, filename = batch
        seg, querys = out

        loss_res = 0
        
        for loss_name, f in self.loss_fn.items():
            if loss_name == "segment_loss":
                loss = f(seg, annotations)
            # elif loss_name == "edge_loss":
            #     max_dim = 100
            #     fea = querys[0]
            #     fea = F.normalize(fea, dim=1)
            #     fea = einops.rearrange(fea, 'b c h w -> (b h w) c')
            #     edge_map = self.edge_conv2d(annotations)
            #     vutils.save_image(edge_map[0].unsqueeze(dim=0).float() / edge_map.max(), '/home/lib/generate_seg/output/test.png')
            #     edge_map = einops.rearrange(edge_map, 'b h w -> (b h w)')
            #     background = fea[edge_map==1, :]
            #     background_nc = background[:max_dim] if background.shape[0] > max_dim else background
            #     edge = fea[edge_map==2, :]
            #     edge_nc = edge[:max_dim] if edge.shape[0] > max_dim else edge
            #     edge_p = edge_nc.mean(dim=0, keepdim=True) # 1 c
            #     edge_c1 = einops.rearrange(edge_p, '1 c -> c 1')
            #     loss = torch.exp(-edge_nc.matmul(edge_c1).abs()).mean() / torch.exp(-background_nc.matmul(edge_c1).abs()).mean()
            #     loss = loss * 0.2
            else:
                continue
            self.log(f'{mode}/{loss_name}', loss)
            loss_res = loss_res + loss
        
        self.log(f'{mode}/loss', loss_res)

        return loss_res
        
    def calculate_metric(self, out, batch, batch_idx, mode):
        images, annotations, labels, filename = batch
        seg, querys = out
        metric_res = {}
        for metric_name, f in self.metric_fn.items():
            if metric_name == "segment_accuracy":
                metric = f(seg, annotations)
            elif metric_name == "segment_f1score":
                metric = f(seg, annotations)
            elif metric_name == "segment_precision":
                metric = f(seg, annotations)
            elif metric_name == "segment_recall":
                metric = f(seg, annotations)
            elif metric_name == "segment_iou":
                metric = f(seg, annotations)
            else:
                continue

            for index in range(len(self.type_classes)):
                t = self.type_classes[index]
                self.log(name=f'{mode}/{metric_name}/{t}', value=metric[index])
            self.log(name=f'{mode}/{metric_name}_total', value=metric.mean())

            metric_res[metric_name] = metric
        
        return metric_res

    def training_step(self, batch, batch_idx):
        
        images, annotations, labels, filename = batch
        out = self(images)
        loss = self.calculate_loss(out, batch, batch_idx, 'train')

        return {'loss': loss,
                'output': out}

    def validation_step(self, batch, batch_idx):
        
        images, annotations, labels, filename = batch
        out = self(images)
        
        loss = self.calculate_loss(out, batch, batch_idx, 'val')
        metric = self.calculate_metric(out, batch, batch_idx, 'val')

        return {'loss': loss,
                'output': out}

    def test_step(self, batch, batch_idx):

        images, annotations, labels, filename = batch
        out = self(images)
        
        loss = self.calculate_loss(out, batch, batch_idx, 'test')
        metric = self.calculate_metric(out, batch, batch_idx, 'test')

        return {'loss': loss,
                'output': out}
    
    def log_images(self, batch, outputs):

        images, annotations, labels, filename = batch
        preds, querys = outputs["output"]
        preds = preds.argmax(dim=1)
        
        ret = {}

        n = min(2, len(filename))

        for i in range(n):
            name = filename[i]
            image = (images[i] + 1.) / 2.
            label_color = self.map_color(annotations[i])
            pred_color = self.map_color(preds[i])
            ret[name] = torch.stack([image, label_color, pred_color], dim=0)

        return ret

    def map_color(self, annotation):
        '''
        input: [HW]
        '''
        annotation = self.color_map.to(self.device)[annotation]
        return einops.rearrange(annotation, 'h w c -> c h w')