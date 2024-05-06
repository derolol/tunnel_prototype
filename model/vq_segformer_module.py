# Ref:
# [1] Code: https://github.com/miracleyoo/pytorch-lightning-template
# [2] Version1: fea0 feature_backgound x feature_defect_edge

import numpy as np
import einops
from omegaconf import OmegaConf
from munch import DefaultMunch

import torch
from torch.nn import Conv2d
from torch.nn import functional as F
from torch.nn import MSELoss
from torch.optim import Adam
from torch.optim.lr_scheduler import OneCycleLR, LinearLR

from pytorch_lightning import LightningModule

from model.net.vq_segformer import SegFormer
from model.vqgan_module import ModelModule as VQGANModule
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

        self.vqgan = VQGANModule.load_from_checkpoint('/home/lib/generate_seg/train_viz/tunnel_reconstruct/version_1/checkpoints/step=10000.ckpt')
        self.vqgan.eval()
        self.segformer = SegFormer(num_classes=3,
                                   pretrained_path='model/weight/segformer_b0_backbone_weights.pth',
                                   in_channels=3,
                                   embed_dims=[32, 64, 160, 256],
                                   num_heads=[1, 2, 5, 8],
                                   mlp_ratios=[4, 4, 4, 4],
                                   qkv_bias=True,
                                   depths=[2, 2, 2, 2],
                                   sr_ratios=[8, 4, 2, 1],
                                   drop_rate=0.0,
                                   drop_path_rate=0.1,
                                   head_embed_dim=256)
        
        self.configure_losses()
        self.configure_metics()
    
    def configure_losses(self):
        self.loss_fn = {
            'segment_loss': FocalLoss(),
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

        opt_net = torch.optim.Adam(self.segformer.parameters(), lr=self.learning_rate)

        scheduler_net = OneCycleLR(optimizer=opt_net,
                                   max_lr=self.learning_rate,
                                   total_steps=self.trainer.max_steps)

        return [opt_net], [scheduler_net]

    def forward(self, img):
        with torch.no_grad():
            encoded_images = self.vqgan.vqgan.encoder(img)
            quant_conv_encoded_images = self.vqgan.vqgan.quant_conv(encoded_images)
            codebook_mapping, codebook_indices, q_loss = self.vqgan.vqgan.codebook(quant_conv_encoded_images)
            post_quant_conv_mapping = self.vqgan.vqgan.post_quant_conv(codebook_mapping)

        output = self.segformer(img, post_quant_conv_mapping)

        return output
    
    def log(self, name, value):
        super().log(name=name,
                    value=value,
                    on_step=True,
                    on_epoch=True,
                    prog_bar=True,
                    sync_dist=True,
                    logger=True)
        
    def calculate_loss(self, out, batch, batch_idx, mode):

        images, annotations, labels, filename = batch
        seg = out

        loss_res = 0

        for loss_name, f in self.loss_fn.items():
            if loss_name == "segment_loss":
                loss = f(seg, annotations)
            else:
                continue

            self.log(f'{mode}/{loss_name}', loss)
            
            loss_res = loss_res + loss
        
        self.log(f'{mode}/loss', loss_res)

        return loss_res
        
    def calculate_metric(self, out, batch, batch_idx, mode):
        images, annotations, labels, filename = batch
        seg = out
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
        preds = outputs["output"]
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