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

from model.net.segformer import SegFormer
from model.net.vqgan import VQGAN
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

        args_dict = {
            "learning_rate": 2.25e-05,
            "latent_dim": 256,
            "image_size": 512,
            "num_codebook_vectors": 1024,
            "beta": 0.25,
            "image_channels": 3,
            "beta1": 0.5,
            "beta2": 0.9,
            "disc_start": 10000,
            "disc_factor": 1.0,
            "rec_loss_factor": 1.0,
            "perceptual_loss_factor": 1.0,
        }

        args = DefaultMunch.fromDict(args_dict)

        self.args = args

        self.vqgan = VQGAN(self.args)
        vqgan_weight = torch.load("/home/lib/generate_seg/train_viz/tunnel_patch_vqgan/version_4/checkpoints/step=100000.ckpt")["state_dict"]
        vqgan_new_weight = {}
        # Get VQGAN weight
        for key, value in vqgan_weight.items():
            if "vqgan." not in key:
                continue
            vqgan_new_weight[key.replace("vqgan.", "")] = value
        self.vqgan.load_state_dict(vqgan_new_weight)
        # Frozen weight
        for name, parameter in self.vqgan.named_parameters():
            if 'decoder' not in name and 'codebook' not in name:
                parameter.requires_grad = False

        self.segformer = SegFormer(num_classes=3,
                                   pretrained_path='model/weight/segformer_b0_backbone_weights.pth',
                                   in_channels=6,
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

        self.automatic_optimization = False
    
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

        lr = self.args.learning_rate

        opt_vq = torch.optim.Adam(
            list(self.vqgan.codebook.parameters()) +
            list(self.vqgan.decoder.parameters()),
            lr=lr, eps=1e-08, betas=(self.args.beta1, self.args.beta2)
        )
        opt_seg = torch.optim.Adam(self.segformer.parameters(), lr=self.learning_rate)

        scheduler_vq = OneCycleLR(optimizer=opt_vq,
                                   max_lr=lr,
                                   total_steps=self.trainer.max_steps)

        scheduler_seg = OneCycleLR(optimizer=opt_seg,
                                   max_lr=self.learning_rate,
                                   total_steps=self.trainer.max_steps)

        return [opt_vq, opt_seg], [scheduler_vq, scheduler_seg]

    def forward(self, img):
        

        with torch.no_grad():
            encoded_images = self.vqgan.encoder(img)
            quant_conv_encoded_images = self.vqgan.quant_conv(encoded_images)

        codebook_mapping, codebook_indices, q_loss = self.vqgan.codebook(quant_conv_encoded_images)
        post_quant_conv_mapping = self.vqgan.post_quant_conv(codebook_mapping)
        reconstruct_images = self.vqgan.decoder(post_quant_conv_mapping)

        concat_images = torch.concat([img, reconstruct_images], dim=1)

        seg = self.segformer(concat_images)

        return {"reconstruct_images": reconstruct_images,
                "seg": seg}
    
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
        seg = out["seg"]

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
        seg = out["seg"]
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
        
        opt_vq, opt_seg = self.optimizers()
        sch_vq, sch_seg = self.lr_schedulers()

        images, annotations, labels, filename = batch
        out = self(images)
        loss = self.calculate_loss(out, batch, batch_idx, 'train')

        opt_vq.zero_grad()
        opt_seg.zero_grad()
        self.manual_backward(loss)
        opt_vq.step()
        opt_seg.step()
        sch_seg.step()

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
        reconstruct_images = outputs["output"]["reconstruct_images"]
        preds = outputs["output"]["seg"]
        preds = preds.argmax(dim=1)

        ret = {}

        n = min(2, len(filename))

        for i in range(n):
            name = filename[i]
            image = (images[i] + 1.) / 2.
            reconstruct_image = (reconstruct_images[i] + 1.) / 2.
            label_color = self.map_color(annotations[i])
            pred_color = self.map_color(preds[i])
            ret[name] = torch.stack([image, reconstruct_image, label_color, pred_color], dim=0)

        return ret

    def map_color(self, annotation):
        '''
        input: [HW]
        '''
        annotation = self.color_map.to(self.device)[annotation]
        return einops.rearrange(annotation, 'h w c -> c h w')