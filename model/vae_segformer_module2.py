# Ref:
# [1] Code: https://github.com/miracleyoo/pytorch-lightning-template
# [2] Version1: fea0 feature_backgound x feature_defect_edge

import numpy as np
import einops
from omegaconf import OmegaConf

import torch
from torch.nn import Conv2d
from torch.nn import functional as F
from torch.nn import MSELoss
from torch.optim import Adam
from torch.optim.lr_scheduler import OneCycleLR, LinearLR,CosineAnnealingLR

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
            load_state_dict(model=self.model,
                            state_dict=torch.load(resume, map_location="cpu"),
                            strict=True)
        
        self.configure_losses()
        self.configure_metics()

        self.automatic_optimization = False
    
    def configure_losses(self):
        self.loss_fn = {
            'segment_loss': FocalLoss(),
            'reconstruct_loss': MSELoss(),
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

        vae_lr = 0.0001
        min_vae_lr = 0.00001

        self.frozen(self.model.decode_head.reconstruct1)
        opt_net = torch.optim.Adam(filter(lambda p: p.requires_grad, self.model.parameters()), lr=self.learning_rate)
        self.unfrozen(self.model.decode_head.reconstruct1)
        opt_vae = torch.optim.Adam(self.model.decode_head.reconstruct1.parameters(), lr=vae_lr)

        scheduler_net = OneCycleLR(optimizer=opt_net,
                                   max_lr=self.learning_rate,
                                   total_steps=self.trainer.max_steps)
        scheduler_vae = CosineAnnealingLR(optimizer=opt_vae,
                                          T_max=self.trainer.max_steps,
                                          eta_min=min_vae_lr)
        # scheduler_vae = LinearLR(optimizer=opt_vae,
        #                          start_factor=vae_lr,
        #                          end_factor=min_vae_lr,
        #                          total_iters=self.trainer.max_steps)

        return [opt_net, opt_vae], [scheduler_net, scheduler_vae]

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
        
    
    def frozen(self, module):
        for layer in  module.parameters():
            layer.requires_grad = False
    
    def unfrozen(self, module):
        for layer in  module.parameters():
            layer.requires_grad = True

    def calculate_loss(self, out, batch, batch_idx, mode):

        if mode == "train":
            opt_net, opt_vae = self.optimizers()
            sch_net, sch_vae = self.lr_schedulers()

        images, annotations, labels, filename = batch
        hiddens, recons, seg = out

        loss_res = 0

        for loss_name, f in self.loss_fn.items():
            if loss_name == "segment_loss":
                loss = f(seg, annotations)
                if mode == "train":
                    self.frozen(self.model.decode_head.reconstruct1)
                    opt_net.zero_grad()
                    self.manual_backward(loss)
                    opt_net.step()
                    sch_net.step()
                    self.unfrozen(self.model.decode_head.reconstruct1)

            elif loss_name == "reconstruct_loss":
                
                src_fea, reconstruct1, mean, logvar = recons

                # Generate prototype
                with torch.no_grad():
                    B_fea, C_fea, H_fea, W_fea = src_fea.shape
                    
                    patch_annotations = F.max_pool2d(annotations.float(), kernel_size=7, stride=4, padding=3).long()
                    patch_annotations = einops.rearrange(patch_annotations, 'b h w -> (b h w)')
                    
                    patch_background_weight = F.max_pool2d(seg[:, 0 : 1], kernel_size=7, stride=4, padding=3)
                    patch_background_weight = einops.rearrange(patch_background_weight, 'b c h w -> (b h w) c')
                    patch_crack_weight = F.max_pool2d(seg[:, 1 : 2], kernel_size=7, stride=4, padding=3)
                    patch_crack_weight = einops.rearrange(patch_crack_weight, 'b c h w -> (b h w) c')
                    patch_tile_weight = F.max_pool2d(seg[:, 2 : 3], kernel_size=7, stride=4, padding=3)
                    patch_tile_weight = einops.rearrange(patch_tile_weight, 'b c h w -> (b h w) c')
                    
                    patch_feature = einops.rearrange(src_fea, 'b c h w -> (b h w) c')
                    
                    background_feature = patch_feature * patch_background_weight
                    background_feature = background_feature[patch_annotations==0]
                    crack_feature = patch_feature * patch_crack_weight
                    crack_feature = crack_feature[patch_annotations==1, :]
                    tile_feature = patch_feature * patch_tile_weight
                    tile_feature = tile_feature[patch_annotations==2, :]

                    property_feature = torch.zeros_like(patch_feature).to(self.device)
                    if background_feature.nelement() > 0:
                        property_feature[patch_annotations==0] = background_feature.mean(dim=0)
                    if crack_feature.nelement() > 0:
                        property_feature[patch_annotations==1] = crack_feature.mean(dim=0)
                    if tile_feature.nelement() > 0:
                        property_feature[patch_annotations==2] = tile_feature.mean(dim=0)
                    property_feature = einops.rearrange(property_feature, '(b h w) c -> b c h w', b=B_fea, h=H_fea, w=W_fea)

                recon_loss = f(reconstruct1, property_feature)
                # kl_loss = 0.5 * (mean.pow(2) + logvar.exp() - logvar - 1).sum(dim=1)
                # kl_loss = torch.mean(kl_loss)
                # loss = recon_loss + kl_loss
                loss = recon_loss * 0.1

                if mode == "train":
                    opt_vae.zero_grad()
                    self.manual_backward(loss)
                    opt_vae.step()
                    sch_vae.step()
            else:
                continue

            self.log(f'{mode}/{loss_name}', loss)
            
            loss_res = loss_res + loss
        
        self.log(f'{mode}/loss', loss_res)

        return loss_res
        
    def calculate_metric(self, out, batch, batch_idx, mode):
        images, annotations, labels, filename = batch
        _, recons, seg = out
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
        _, _, preds = outputs["output"]
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