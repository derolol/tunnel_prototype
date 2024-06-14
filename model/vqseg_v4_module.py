import random
from lpips import LPIPS
from munch import DefaultMunch

import torch
from torch import nn
from torch.nn import functional as F

from pytorch_lightning import LightningModule

from model.net.vqseg_v4 import VQSeg, Discriminator

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

        self.learning_rate = learning_rate
        self.num_classes = num_classes
        self.type_classes = type_classes
        self.color_map = torch.tensor(color_map)
        self.beta1 = 0.9    # 0.5
        self.beta2 = 0.999  #0.9

        self.num_vectors = 128

        self.vqseg = VQSeg(num_classes=num_classes,
                           in_channels=3,
                           embed_dims=[32, 64, 128, 256],
                           num_heads=[1, 2, 4, 8],
                           mlp_ratios=[4, 4, 4, 4],
                           qkv_bias=True,
                           depths=[2, 2, 2, 2],
                           sr_ratios=[8, 4, 2, 1],
                           drop_rate=0.,
                           latent_dim=256,
                           num_vectors=self.num_vectors,
                           beta=0.25)
        
        self.discriminator = Discriminator(image_channels=3)

        self.configure_losses()
        self.configure_metics()

        self.automatic_optimization = False
    
    def configure_losses(self):
        self.loss_fn = {
            'reconstruct_loss': 0,
            'quant_loss': 0,
            'disc_loss': 0,
        }
    
    def configure_metics(self):
        self.metric_fn = {
        }

    def forward(self, img):
        return self.vqseg(img)

    def configure_optimizers(self):

        opt_vq = torch.optim.Adam(
            self.vqseg.parameters(),
            lr=self.learning_rate, eps=1e-08, betas=(self.beta1, self.beta2)
        )

        opt_disc = torch.optim.Adam(self.discriminator.parameters(),
                                    lr=self.learning_rate)

        return [opt_vq, opt_disc]

    def log(self, name, value):
        super().log(name=name,
                    value=value,
                    on_step=True,
                    on_epoch=True,
                    prog_bar=True,
                    sync_dist=True,
                    logger=True)
        
    def calculate_loss(self, out, batch, batch_idx, mode):
        
        if mode == "train":
            opt_vq, opt_disc = self.optimizers()

        images, annotations, labels, filename = batch
        decoded_images, _, quant_loss, quant_disc, points = self.vqseg(images)

        loss_res = 0
        
        for loss_name, f in self.loss_fn.items():
            if loss_name == "reconstruct_loss":
                disc_real = self.discriminator(images)
                disc_fake = self.discriminator(decoded_images)
                fake_loss = 0
                if self.global_step > 2000:
                    fake_loss = - torch.mean(disc_fake)
                color_loss = (1 - (images * decoded_images).sum(dim=1)).mean()
                rec_loss = torch.abs(images - decoded_images).mean()
                loss = fake_loss + rec_loss + color_loss
            elif loss_name == "quant_loss":
                loss = quant_loss
            elif loss_name == 'disc_loss':
                disc_real = self.discriminator(images)
                disc_fake = self.discriminator(decoded_images.detach())
                d_loss_real = torch.mean(F.relu(1. - disc_real))
                d_loss_fake = torch.mean(F.relu(1. + disc_fake))
                disc_loss = d_loss_real + d_loss_fake
                self.log(f'{mode}/disc_loss', disc_loss)
                continue
            else:
                continue

            self.log(f'{mode}/{loss_name}', loss)
            
            loss_res = loss_res + loss
        
        self.log(f'{mode}/loss', loss_res)

        if mode == 'train':
            opt_vq.zero_grad()
            self.manual_backward(loss)

            opt_disc.zero_grad()
            self.manual_backward(disc_loss)

            opt_vq.step()
            opt_disc.step()

        return loss_res
    
    def calculate_metric(self, out, batch, batch_idx, mode):
        
        images, annotations, labels, filename = batch
        decoded_images, _, quant_loss, quant_disc, points = self.vqseg(images)
        
        metric_res = {}
        for metric_name, f in self.metric_fn.items():
            if metric_name == "metric":
                metric = f(decoded_images, images)
            else:
                continue

            self.log(name=f'{mode}/{metric_name}', value=metric.mean())

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
        decoded_images, _, _, _, _ = outputs["output"]

        ret = {}

        n = min(4, len(filename))

        for i in range(n):
            name = filename[i]
            image = (images[i] + 1.) / 2.
            decoded_image = (decoded_images[i] + 1.) / 2.
            ret[name] = torch.stack([image, decoded_image], dim=0)

        return ret
