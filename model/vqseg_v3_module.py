# Ref:
# [1] Code: https://github.com/miracleyoo/pytorch-lightning-template
# [2] Version1: fea0 feature_backgound x feature_defect_edge

import einops

import torch
from torch.optim import Adam
from torch.optim.lr_scheduler import OneCycleLR, LinearLR
from torch.nn import MSELoss, CrossEntropyLoss
from torchvision.models.vgg import VGG
from torchvision.models import vgg16_bn, VGG16_BN_Weights

from pytorch_lightning import LightningModule

from model.loss.focal_loss import FocalLoss
from model.metric.seg_metric import SegAccuracyMetric, SegFBetaScoreMetric, SegPrecisionMetric, SegRecallMetric, SegIoUMetric
from model.net.vqseg_v3 import VQSeg

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
    
        self.vgg = vgg16_bn(weights=VGG16_BN_Weights)
        for param in self.vgg.parameters():
            param.requires_grad = False
        self.vgg.eval()
        
        self.configure_losses()
        self.configure_metics()

    def configure_losses(self):
        self.loss_fn = {
            'segment_loss': FocalLoss(),
            'quant_loss': 0,
            'disc_loss': CrossEntropyLoss(),
            'point_loss': CrossEntropyLoss(ignore_index=0),
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

        lr = self.learning_rate

        optimizer = Adam(self.vqseg.parameters(),
                         lr=lr,
                         eps=1e-08,
                         betas=(self.beta1, self.beta2))

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
        return self.vqseg(img)
    
    def log(self, name, value):
        super().log(name=name,
                    value=value,
                    on_step=True,
                    on_epoch=True,
                    prog_bar=True,
                    sync_dist=True,
                    logger=True)
        
    
    def point_sample(self, input_, point_coords, **kwargs):
        add_dim = False
        if point_coords.dim() == 3:
            add_dim = True
            point_coords = point_coords.unsqueeze(2)
        output = torch.nn.functional.grid_sample(input_, 2.0 * point_coords - 1.0, **kwargs)
        if add_dim:
            output = output.squeeze(3)
        return output

    def calculate_loss(self, out, batch, batch_idx, mode):

        images, annotations, labels, filename = batch
        seg, _, quant_loss, quant_disc, points = out

        loss_res = 0

        for loss_name, f in self.loss_fn.items():
            if loss_name == "segment_loss":
                loss = f(seg, annotations)
            elif loss_name == "quant_loss":
                loss = quant_loss * 10
            elif loss_name == 'disc_loss':
                disc_label = annotations.float().unsqueeze(dim=1)
                disc_label = torch.nn.functional.interpolate(disc_label,
                                                             size=quant_disc.shape[2:],
                                                             mode='nearest')
                patch_size = 32
                B, C, H, W = quant_disc.shape
                quant_disc = einops.rearrange(quant_disc,
                                              'b c (n p1) (m p2) -> (b n m) c p1 p2', 
                                              n=H // patch_size,
                                              m=W // patch_size,
                                              p1=patch_size,
                                              p2=patch_size)
                B2, C2, H2, W2 = quant_disc.shape
                pad = torch.zeros((B2, 1, H2, W2)).to(self.device)
                quant_disc = torch.concat([quant_disc, pad], dim=1)

                # print(disc_label.shape)
                disc_label = torch.nn.functional.interpolate(disc_label,
                                                             size=[H // patch_size, W // patch_size],
                                                             mode='nearest')
                # print(disc_label.shape)
                disc_label = disc_label.long().squeeze(dim=1)
                disc_label = einops.rearrange(disc_label,
                                              'b n m -> (b n m)', 
                                              n=H // patch_size,
                                              m=W // patch_size)
                # print(disc_label.shape)
                loss = f(self.vgg(quant_disc), disc_label)
            elif loss_name == 'point_loss':
                gt_points = self.point_sample(
                    annotations.float().unsqueeze(1),
                    points['points'],
                    mode='nearest',
                    align_corners=False
                ).squeeze_(1).long()
                loss = f(points['rend'], gt_points) 
            else:
                continue

            self.log(f'{mode}/{loss_name}', loss)
            
            loss_res = loss_res + loss
        
        self.log(f'{mode}/loss', loss_res)

        return loss_res
        
    def calculate_metric(self, out, batch, batch_idx, mode):
        images, annotations, labels, filename = batch
        seg, _, quant_loss, quant_disc, points = out
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
    
    # def on_train_batch_end(self, outputs, batch, batch_idx) -> None:
        
    #     super().on_train_batch_end(outputs, batch, batch_idx)
    #     self.print("backward:", self.vqseg.codebook.embedding_value.grad)

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
        preds, _, _, _, _ = outputs["output"]
        preds = preds.argmax(dim=1)

        ret = {}

        n = min(4, len(filename))

        begin = torch.tensor([1., 0., 0.]).to(self.device).unsqueeze(dim=-1).unsqueeze(dim=-1)
        end = torch.tensor([0., 0., 1.]).to(self.device).unsqueeze(dim=-1).unsqueeze(dim=-1)
        inter = end - begin

        embedding_key = self.vqseg.codebook.embedding_key.weight.detach()
        sim_key = embedding_key @ embedding_key.t() # [num_vector, num_vector]
        sim_key = (sim_key  + 1.) / 2.
        sim_key = sim_key.unsqueeze(dim=0).repeat([3, 1, 1])
        sim_key = torch.nn.functional.interpolate(sim_key.unsqueeze(dim=0), size=images.shape[2:]).squeeze(dim=0)
        sim_key = sim_key * inter + begin
        self.print("sim_key", sim_key.mean(), sim_key.median(), sim_key.max(), sim_key.min())
        
        embedding_value = self.vqseg.codebook.embedding_value.detach()
        sim_value = embedding_value @ embedding_value.t() # [num_vector, num_vector]
        sim_value = (sim_value  + 1.) / 2.
        sim_value = sim_value.unsqueeze(dim=0).repeat([3, 1, 1])
        sim_value = torch.nn.functional.interpolate(sim_value.unsqueeze(dim=0), size=images.shape[2:]).squeeze(dim=0)
        sim_value = sim_value * inter + begin
        self.print("sim_value", sim_value.mean(), sim_value.median(), sim_value.max(), sim_value.min())

        for i in range(n):
            name = filename[i]
            image = (images[i] + 1.) / 2.
            label_color = self.map_color(annotations[i])
            pred_color = self.map_color(preds[i])
            ret[name] = torch.stack([image, label_color, pred_color, sim_key, sim_value], dim=0)

        return ret

    def map_color(self, annotation):
        '''
        input: [HW]
        '''
        annotation = self.color_map.to(self.device)[annotation]
        return einops.rearrange(annotation, 'h w c -> c h w')