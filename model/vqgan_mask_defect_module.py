import random
from lpips import LPIPS
from munch import DefaultMunch

import torch
from torch import nn
from torch.nn import functional as F

from pytorch_lightning import LightningModule

from model.net.vqgan import Discriminator
from model.net.vqgan import VQGAN

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)

class ModelModule(LightningModule):

    def __init__(self,
                #  learning_rate: float = 2.25e-05,
                #  latent_dim: int = 256,
                #  image_size: int = 512,
                #  num_codebook_vectors: int = 1024,
                #  beta: float = 0.25,
                #  image_channels: int = 3,
                #  beta1: float = 0.5,
                #  beta2: float = 0.9,
                #  disc_start: int = 10000,
                #  disc_factor: float = 1.0,
                #  rec_loss_factor: float = 1.0,
                #  perceptual_loss_factor: float = 1.0,
                 *args, **kwargs) -> None:
        
        super().__init__()

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
        self.discriminator = Discriminator(image_channels=3)
        self.discriminator.apply(weights_init)

        self.configure_losses()

        self.automatic_optimization = False
    
    def configure_losses(self):
        self.loss_fn = {
            'perceptual_loss': LPIPS().eval(),
        }

    def configure_optimizers(self):

        lr = self.args.learning_rate

        opt_vq = torch.optim.Adam(
            list(self.vqgan.encoder.parameters()) +
            list(self.vqgan.decoder.parameters()) +
            list(self.vqgan.codebook.parameters()) +
            list(self.vqgan.quant_conv.parameters()) +
            list(self.vqgan.post_quant_conv.parameters()),
            lr=lr, eps=1e-08, betas=(self.args.beta1, self.args.beta2)
        )

        opt_disc = torch.optim.Adam(self.discriminator.parameters(),
                                    lr=lr,
                                    eps=1e-08,
                                    betas=(self.args.beta1, self.args.beta2))

        return [opt_vq, opt_disc]

    def log(self, name, value):
        super().log(name=name,
                    value=value,
                    on_step=True,
                    on_epoch=True,
                    prog_bar=True,
                    sync_dist=True,
                    logger=True)

    def forward(self, batch, mode):

        opt_vq, opt_disc = self.optimizers()
        images, annotations, labels, filename = batch

        decoded_images, _, q_loss = self.vqgan(images)

        if mode == "train":

            # Local mask 
            local_mask = F.max_pool2d(annotations.float(), kernel_size=7, padding=3, stride=1)
            local_mask[local_mask > 0] = 1.0
            local_mask = local_mask.unsqueeze(dim=1).repeat(repeats=(1, 3, 1, 1))
            local_mask_bincount = local_mask.long().flatten().bincount().numpy()
            local_mask_num = 0 if len(local_mask_bincount) == 1 else local_mask_bincount[1]
            assert local_mask.shape == images.shape

            mask_images = images * local_mask
            mask_decoded_images = decoded_images * local_mask

            # Global disc
            disc_real = self.discriminator(images)
            disc_fake = self.discriminator(decoded_images)

            # Local disc
            disc_local_real = self.discriminator(mask_images)
            disc_local_fake = self.discriminator(mask_decoded_images)

            # If start disc
            disc_factor = self.vqgan.adopt_weight(self.args.disc_factor,
                                                self.global_step,
                                                threshold=self.args.disc_start)

            # Global reconstruct
            perceptual_loss = self.loss_fn["perceptual_loss"].to(self.device)(images, decoded_images)
            rec_loss = torch.abs(images - decoded_images)
            perceptual_rec_loss = self.args.perceptual_loss_factor * perceptual_loss + self.args.rec_loss_factor * rec_loss
            perceptual_rec_loss = perceptual_rec_loss.mean()
            self.log(f'{mode}/perceptual_rec_loss', perceptual_rec_loss)

            # Local reconstruct
            perceptual_local_loss = self.loss_fn["perceptual_loss"].to(self.device)(mask_images, mask_decoded_images)
            rec_local_loss = torch.abs(mask_images - mask_decoded_images)
            perceptual_rec_local_loss = self.args.perceptual_loss_factor * perceptual_local_loss + self.args.rec_loss_factor * rec_local_loss
            # Defect number
            # perceptual_rec_local_loss = perceptual_rec_local_loss.mean()
            perceptual_rec_local_loss = perceptual_rec_local_loss / local_mask_num
            self.log(f'{mode}/perceptual_rec_local_loss', perceptual_rec_local_loss)

            # Global & Local VQ loss
            g_loss = - torch.mean(disc_fake)
            λ = self.vqgan.calculate_lambda(perceptual_rec_loss, g_loss)
            g_local_loss = - torch.mean(disc_local_fake)
            λ_local = self.vqgan.calculate_lambda(perceptual_rec_local_loss, g_local_loss)
            vq_loss = perceptual_rec_loss + perceptual_rec_local_loss + q_loss + disc_factor * λ * g_loss + disc_factor * λ_local * g_local_loss
            self.log(f'{mode}/vq_loss', vq_loss)
            
            # Global GAN loss
            d_loss_real = torch.mean(F.relu(1. - disc_real))
            d_loss_fake = torch.mean(F.relu(1. + disc_fake))
            gan_global_loss = disc_factor * 0.5 * (d_loss_real + d_loss_fake)
            self.log(f'{mode}/gan_global_loss', gan_global_loss)

            # Local GAN loss
            d_local_loss_real = torch.mean(F.relu(1. - disc_local_real))
            d_local_loss_fake = torch.mean(F.relu(1. + disc_local_fake))
            gan_local_loss = disc_factor * 0.5 * (d_local_loss_real + d_local_loss_fake)
            self.log(f'{mode}/gan_local_loss', gan_local_loss)

            gan_loss = gan_global_loss + gan_local_loss

            opt_vq.zero_grad()
            self.manual_backward(vq_loss, retain_graph=True)

            opt_disc.zero_grad()
            self.manual_backward(gan_loss)

            opt_vq.step()
            opt_disc.step()

        return {"perceptual_rec_loss": perceptual_rec_loss if mode == "train" else 0,
                "gan_loss": gan_loss if mode == "train" else 0,
                "decoded_images": decoded_images,
                "mask_images": mask_images if mode == "train" else None,
                "mask_decoded_images": mask_decoded_images if mode == "train" else None}
        
    def training_step(self, batch, batch_idx):
        
        output = self(batch, 'train')
        return output

    def validation_step(self, batch, batch_idx):
        
        output = self(batch, 'val')
        return output

    def test_step(self, batch, batch_idx):

        output = self(batch, 'test')
        return output
    
    def log_images(self, batch, outputs):

        images, annotations, labels, filename = batch

        decoded_images = outputs["decoded_images"]
        mask_images = outputs["mask_images"]
        mask_decoded_images = outputs["mask_decoded_images"]

        ret = {}

        n = min(4, len(filename))

        for i in range(n):
            name = filename[i]
            image = (images[i] + 1.) / 2.
            decoded_image = (decoded_images[i] + 1.) / 2.
            mask_image = (mask_images[i] + 1.) / 2.
            mask_decoded_image = (mask_decoded_images[i] + 1.) / 2.
            ret[name] = torch.stack([image, decoded_image, mask_image, mask_decoded_image], dim=0)

        return ret
