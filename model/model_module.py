# Ref:
# [1] Code: https://github.com/miracleyoo/pytorch-lightning-template
# [2] Code: https://github.com/XPixelGroup/DiffBIR

import einops
from omegaconf import OmegaConf

import torch
from torch.nn import functional as F
from torch.optim import Adam
from torch.optim.lr_scheduler import OneCycleLR

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
    
    def configure_losses(self):
        self.loss_fn = {
            'segment_loss': FocalLoss()
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

    def calculate_loss(self, out, batch, batch_idx, mode):
        images, annotations, labels, filename = batch
        loss_res = 0
        for loss_name, f in self.loss_fn.items():
            if loss_name == "segment_loss":
                loss = f(out, annotations)
            else:
                continue
            self.log(f'{mode}/{loss_name}', loss)
            loss_res = loss_res + loss
        self.log(f'{mode}/loss', loss_res)

        return loss_res
        
    def calculate_metric(self, out, batch, batch_idx, mode):
        images, annotations, labels, filename = batch
        metric_res = {}
        for metric_name, f in self.metric_fn.items():
            if metric_name == "segment_accuracy":
                metric = f(out, annotations)
            elif metric_name == "segment_f1score":
                metric = f(out, annotations)
            elif metric_name == "segment_precision":
                metric = f(out, annotations)
            elif metric_name == "segment_recall":
                metric = f(out, annotations)
            elif metric_name == "segment_iou":
                metric = f(out, annotations)
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

        n = min(4, len(filename))

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
    
    def test_all(self, batch, batch_idx):
        
        images, annotations, filename = batch
        B = images.shape[0]
        annotations = annotations.squeeze(dim=1)
        out = self(images)

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