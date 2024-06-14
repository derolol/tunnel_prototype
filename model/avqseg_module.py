import einops
import numpy as np

import torch
from torch import nn
from torch.optim import Adam
from torch.optim.lr_scheduler import OneCycleLR, LinearLR
from torch.nn import CrossEntropyLoss, MSELoss
from torch import fft

from pytorch_lightning import LightningModule

from model.loss.focal_loss import FocalLoss
from model.metric.seg_metric import SegAccuracyMetric, SegFBetaScoreMetric, SegPrecisionMetric, SegRecallMetric, SegIoUMetric

from model.net.avqseg import AVQSeg

class QuantLoss(nn.Module):

    def __init__(self, beta, embedding_frozen) -> None:
        
        super().__init__()

        self.beta = beta
        self.embedding_frozen = embedding_frozen

    def forward(self, x_q, x_fea):
        
        if self.embedding_frozen:
            loss_fea = 0
        else:
            loss_fea = torch.mean((x_q.detach() - x_fea) ** 2)
        loss_q = self.beta * torch.mean((x_q - x_fea.detach()) ** 2)

        loss = loss_fea + loss_q

        return loss

class FFTLoss(nn.Module):

    def __init__(self, input_size=512, radius=16) -> None:
        
        super().__init__()

        self.loss = MSELoss()

        center = (int(input_size / 2), int(input_size / 2))

        Y, X = np.ogrid[ : input_size, : input_size]
        dist_from_center = np.sqrt((X - center[0]) ** 2 + (Y-center[1]) ** 2)
        mask = dist_from_center <= radius

        self.mask = torch.tensor(mask, dtype=torch.float)
        self.mask = self.to_complex(self.mask)
    
    def to_complex(self, t):
        i = torch.zeros_like(t).to(t)
        return torch.complex(t, i)

    def forward(self, pred, annotation):

        self.mask = self.mask.to(pred.device)
        pred = pred.softmax(dim=1)
        pred = pred.argmax(dim=1)
        # pred = pred[:, 1] - pred[:, 0]
        B = pred.shape[0]

        preds = torch.split(pred.float(), split_size_or_sections=B, dim=0)
        annotations = torch.split(annotation.float(), split_size_or_sections=B, dim=0)

        # pred_fft = torch.stack([fft.fft2(self.to_complex(p) * self.mask).real.float() for p in preds], dim=0)
        # mask_fft = torch.stack([fft.fft2(self.to_complex(a) * self.mask).real.float() for a in annotations], dim=0)

        pred_fft = torch.stack([fft.fft2(self.to_complex(p) * self.mask) for p in preds], dim=0)
        mask_fft = torch.stack([fft.fft2(self.to_complex(a) * self.mask) for a in annotations], dim=0)
        
        # |X(f)| = sqrt(real(X(f))^2 + imag(X(f))^2)
        pred_len = (pred_fft.real ** 2 + pred_fft.imag ** 2).sqrt()
        mask_len = (mask_fft.real ** 2 + mask_fft.imag ** 2).sqrt()
        # angle(X(f)) = atan2(imag(X(f)), real(X(f)))
        pre_angle = torch.atan2(pred_fft.imag, pred_fft.real)
        mask_angle = torch.atan2(mask_fft.imag, mask_fft.real)

        return self.loss(pred_len, mask_len) + self.loss(pre_angle, mask_angle)

class ModelModule(LightningModule):

    def __init__(self,
                 learning_rate, # model
                 quant_loss_beta,
                 num_classes, # seg
                 type_classes,
                 color_map,
                 in_channels, # encoder
                 embed_dims,
                 depths,
                 num_heads,
                 mlp_ratios,
                 qkv_bias,
                 sr_ratios,
                 drop_rate,
                 pooling_module,
                 decode_dim, # decoder
                 is_feature_quant, # vector quant
                 embedding_frozen,
                 quant_fuse_module,
                 quant_dim,
                 num_vectors,
                 latent_dim,
                 is_seg_refine,
                 sample_size,
                 is_edge_weight,
                 sample_beta) -> None:
        
        super().__init__()

        self.learning_rate = learning_rate
        self.quant_loss_beta = quant_loss_beta

        self.num_classes = num_classes
        self.type_classes = type_classes
        self.color_map = torch.tensor(color_map)

        self.is_feature_quant = is_feature_quant
        self.embedding_frozen = embedding_frozen
        
        self.is_seg_refine = is_seg_refine
        self.sample_size = sample_size

        self.avqseg = AVQSeg(num_classes=num_classes, # seg
                             in_channels=in_channels, # encoder
                             embed_dims=embed_dims,
                             depths=depths,
                             num_heads=num_heads,
                             mlp_ratios=mlp_ratios,
                             qkv_bias=qkv_bias,
                             sr_ratios=sr_ratios,
                             drop_rate=drop_rate,
                             pooling_module=pooling_module,
                             decode_dim=decode_dim, # decoder
                             is_feature_quant=is_feature_quant, # vector quant
                             embedding_frozen=embedding_frozen,
                             quant_fuse_module=quant_fuse_module,
                             quant_dim=quant_dim,
                             num_vectors=num_vectors,
                             latent_dim=latent_dim,
                             is_seg_refine=is_seg_refine, # refine seg
                             sample_size=sample_size,
                             is_edge_weight=is_edge_weight,
                             sample_beta=sample_beta
                             )
    
        # if is_seg_refine:
        #     from_state_dict = torch.load(f='/home/lib/generate_seg/train_crack500_viz/crack500_avqseg_soft_quant_s/version_1/checkpoints/step=18000.ckpt')['state_dict']
        #     new_state_dict = {}
        #     model_state_dict = self.avqseg.state_dict()
        #     for key, value in model_state_dict.items():
        #         from_key = f'avqseg.{key}'
        #         new_state_dict[key] = from_state_dict.get(from_key, value)
        #     self.avqseg.load_state_dict(state_dict=new_state_dict)

        self.configure_losses()
        self.configure_metics()

    def configure_losses(self):
        self.loss_fn = {
            'segment_loss': FocalLoss(),
            'quant_loss': QuantLoss(beta=self.quant_loss_beta,
                                    embedding_frozen=self.embedding_frozen),
            # 'point_loss': CrossEntropyLoss(),
            # 'point_loss': FocalLoss(),
            'fft_loss': FFTLoss(),
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

        optimizer = Adam(self.avqseg.parameters(),
                         lr=lr,
                         eps=1e-08)

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
        return self.avqseg(img)
    
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

        loss_res = 0

        for loss_name, f in self.loss_fn.items():
            if loss_name == "segment_loss":
                loss = f(out['seg'], annotations)
            elif self.is_feature_quant and loss_name == "quant_loss":
                loss = f(out['x_q'], out['x_fea'])
            # elif out['points'] is not None and self.is_seg_refine and loss_name == 'point_loss':
            #     gt_points = self.avqseg.point_head.point_sample(
            #         annotations.float().unsqueeze(1),
            #         out['points'],
            #         sample_size=self.sample_size,
            #         mode='nearest',
            #         align_corners=False
            #     ).squeeze_(1).long()
            #     out_rend = out['rend']
                # loss = f(out_rend, gt_points) 
                # if self.sample_size > 1:
                #     gt_points = einops.rearrange(gt_points,
                #                                 'b (n s1 s2) -> (b n) s1 s2',
                #                                 s1=self.sample_size,
                #                                 s2=self.sample_size)
                #     out_rend = einops.rearrange(out['rend'],
                #                                 'b c (n s1 s2) -> (b n) c s1 s2',
                #                                 s1=self.sample_size,
                #                                 s2=self.sample_size)
            elif out['origin_seg'] is not None and self.is_seg_refine and loss_name == 'fft_loss':
                loss = f(out['origin_seg'], annotations) * 0.001
            else:
                continue

            self.log(f'{mode}/{loss_name}', loss)
            
            loss_res = loss_res + loss
        
        self.log(f'{mode}/loss', loss_res)

        return loss_res
        
    def calculate_metric(self, out, batch, batch_idx, mode):

        images, annotations, labels, filename = batch
        seg = out['seg']
        
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
        preds = outputs['output']['seg']
        preds = preds.argmax(dim=1)

        ret = {}

        n = min(4, len(filename))

        if self.is_feature_quant:
            begin = torch.tensor([1., 0.8, 0.01]).to(self.device).unsqueeze(dim=-1).unsqueeze(dim=-1)
            end = torch.tensor([0.4, 0.5, 1.]).to(self.device).unsqueeze(dim=-1).unsqueeze(dim=-1)
            inter = end - begin

            embedding = self.avqseg.codebook.embedding.weight.detach()
            embedding = embedding @ embedding.t() # [num_vector, num_vector]
            embedding = (embedding  + 1.) / 2.
            embedding = embedding.unsqueeze(dim=0).repeat([3, 1, 1])
            embedding = torch.nn.functional.interpolate(embedding.unsqueeze(dim=0),
                                                        size=images.shape[2:]).squeeze(dim=0)
            embedding = embedding * inter + begin

        for i in range(n):
            name = filename[i]
            image = (images[i] + 1.) / 2.
            label_color = self.map_color(annotations[i])
            pred_color = self.map_color(preds[i])
            if self.is_feature_quant:
                ret[name] = torch.stack([image, label_color, pred_color, embedding], dim=0)
            else:
                ret[name] = torch.stack([image, label_color, pred_color], dim=0)

        return ret

    def map_color(self, annotation):
        '''
        input: [HW]
        '''
        annotation = self.color_map.to(self.device)[annotation]
        return einops.rearrange(annotation, 'h w c -> c h w')

    def test_all(self, batch, batch_idx):
        
        images, annotations, filename = batch
        B = images.shape[0]
        annotations = annotations.squeeze(dim=1)
        out = self(images)
        out = out['seg']

        rows = []
        for b in range(B):
            precision = self.metric_fn['segment_precision'](out[b : b + 1], annotations[b : b + 1]).cpu().numpy().tolist()
            recall = self.metric_fn['segment_recall'](out[b : b + 1], annotations[b : b + 1]).cpu().numpy().tolist()
            iou = self.metric_fn['segment_iou'](out[b : b + 1], annotations[b : b + 1]).cpu().numpy().tolist()
            f1score = self.metric_fn['segment_f1score'](out[b : b + 1], annotations[b : b + 1]).cpu().numpy().tolist()
            row = [
                filename[b],
                '|'.join([str(i) for i in precision]),
                '|'.join([str(i) for i in recall]),
                '|'.join([str(i) for i in iou]),
                '|'.join([str(i) for i in f1score])
            ]
            rows.append(row)

        return rows, out