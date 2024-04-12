import einops
import numpy as np

import torch
from torch.nn import CrossEntropyLoss, Conv2d
from torch.nn import functional as F
from torch.optim.lr_scheduler import LinearLR
from torchvision import utils as vutils
from pytorch_lightning import LightningModule

from model.metric.seg_metric import SegAccuracyMetric, SegFBetaScoreMetric, SegPrecisionMetric, SegRecallMetric, SegIoUMetric
from model.loss.focal_loss import FocalLoss
from model.net.gan_segformer import SegFormer
from model.net.trans_discriminator import Discriminator

class ModelModule(LightningModule):

    def __init__(
        self,
        model_config,
        resume,
        learning_rate,
        num_classes,
        type_classes,
        color_map,
        **kwargs
    ):
        super().__init__()
        self.save_hyperparameters()

        self.model_config = model_config

        self.learning_rate = learning_rate
        self.num_classes =num_classes
        self.type_classes = type_classes
        self.color_map = torch.tensor(color_map)

        # self.patch_size = 8
        self.patch_size = 16

        # 网络
        self.generator = SegFormer(num_classes= 3,
                                   pretrained_path= 'model/weight/segformer_b0_backbone_weights.pth',
                                   in_channels= 3,
                                   embed_dims= [32, 64, 160, 256],
                                   num_heads= [1, 2, 5, 8],
                                   mlp_ratios= [4, 4, 4, 4],
                                   qkv_bias= True,
                                   depths= [2, 2, 2, 2],
                                   sr_ratios= [8, 4, 2, 1],
                                   drop_rate= 0.0,
                                   drop_path_rate= 0.1,
                                   head_embed_dim=256)
        self.discriminator = Discriminator(img_size=self.patch_size,
                                           in_chans=32,
                                           num_classes=3)
        self.g_loss_weight = 1.0
        self.adversarial_loss = CrossEntropyLoss()
        self.segment_loss = FocalLoss()
        self.configure_metics()

        # Important: This property activates manual optimization.
        self.automatic_optimization = False

    def configure_metics(self):
        self.metric_fn = {
            'segment_accuracy': SegAccuracyMetric(num_classes=self.num_classes),
            'segment_f1score': SegFBetaScoreMetric(num_classes=self.num_classes),
            'segment_precision': SegPrecisionMetric(num_classes=self.num_classes),
            'segment_recall': SegRecallMetric(num_classes=self.num_classes),
            'segment_iou': SegIoUMetric(num_classes=self.num_classes),
        }
    def forward(self, z):
        return self.generator(z)

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

            # edge_detect = F.max_pool2d(edge_detect.unsqueeze(dim=1),
            #                            kernel_size=3,
            #                            stride=2,
            #                            padding=1).squeeze(dim=1)

            return edge_detect
    
    def get_g_patch_valid(self, annotations, c_fea, edge_map):
        '''
        focus on edge
        '''

        B, C, H, W = c_fea.shape
        half_patch_size = self.patch_size // 2

        patch = torch.zeros(size=(B, C, self.patch_size, self.patch_size)).to(self.device)
        valid = torch.zeros(B, ).to(self.device)
        
        for b in range(B):
            edge_map_batch = edge_map[b] # H, W

            edge_index_list = (edge_map_batch==2).nonzero() # N 2
            
            # no defect
            if edge_index_list.shape[0] == 0:
                center_h = np.random.randint(low=0, high=(H - self.patch_size))
                center_w = np.random.randint(low=0, high=(W - self.patch_size))
                patch[b] = c_fea[b,
                                 :,
                                 center_h : center_h + self.patch_size,
                                 center_w : center_w + self.patch_size]
                valid[b] = 0
                continue

            # on edge
            index = edge_index_list[np.random.randint(low=0, high=edge_index_list.shape[0])]
            
            min_x = 0 if index[1] - half_patch_size < 0 else min(index[1] + half_patch_size, W) - self.patch_size
            min_y = 0 if index[0] - half_patch_size < 0 else min(index[0] + half_patch_size, W) - self.patch_size

            patch[b] = c_fea[b,
                           :,
                           min_y : min_y + self.patch_size,
                           min_x : min_x + self.patch_size]
            
            annotation_patch = annotations[b,
                                min_y : min_y + self.patch_size,
                                min_x : min_x + self.patch_size]
            
            num = annotation_patch.flatten().bincount()
            crack_num = 0 if len(num) < 2 else num[1]
            tile_peeling_num = 0 if len(num) < 3 else num[2]

            valid[b] = 1 if crack_num > tile_peeling_num else 2
        
        return patch, valid

    def get_d_patch_valid(self, annotations, c_fea, edge_map):
        '''
        focus on both edge and background
        '''

        B, C, H, W = c_fea.shape
        half_patch_size = self.patch_size // 2

        patch = torch.zeros(size=(B, C, self.patch_size, self.patch_size)).to(self.device)
        valid = torch.zeros(B, ).to(self.device)

        for b in range(B):
            edge_map_batch = edge_map[b] # H, W

            if b < B // 2: # background
                edge_index_list = (edge_map_batch==1).nonzero() # N 2
            else: # edge
                edge_index_list = (edge_map_batch==2).nonzero() # N 2
            
            # not found
            if edge_index_list.shape[0] == 0:
                center_h = np.random.randint(low=0, high=(H - self.patch_size))
                center_w = np.random.randint(low=0, high=(W - self.patch_size))
                patch[b] = c_fea[b,
                                 :,
                                 center_h : center_h + self.patch_size,
                                 center_w : center_w + self.patch_size]
                annotation_patch = annotations[b,
                                center_h : center_h + self.patch_size,
                                center_w : center_w + self.patch_size]
            # found
            else:
                index = edge_index_list[np.random.randint(low=0, high=edge_index_list.shape[0])]
                
                min_x = 0 if index[1] - half_patch_size < 0 else min(index[1] + half_patch_size, W) - self.patch_size
                min_y = 0 if index[0] - half_patch_size < 0 else min(index[0] + half_patch_size, W) - self.patch_size

                patch[b] = c_fea[b,
                                 :,
                                 min_y : min_y + self.patch_size,
                                 min_x : min_x + self.patch_size]
                annotation_patch = annotations[b,
                                 min_y : min_y + self.patch_size,
                                 min_x : min_x + self.patch_size]
            
            # make sure valid
            num = annotation_patch.flatten().bincount()
            if len(num) == 1:
                valid[b] = 0
            else:
                crack_num = 0 if len(num) < 2 else num[1]
                tile_peeling_num = 0 if len(num) < 3 else num[2]
                valid[b] = 1 if crack_num > tile_peeling_num else 2

        return patch, valid

    def training_step(self, batch, batch_idx):

        # Implementation follows the PyTorch tutorial:
        # https://pytorch.org/tutorials/beginner/dcgan_faces_tutorial.html
        g_opt, d_opt = self.optimizers()
        g_sch, d_sch = self.lr_schedulers()

        images, annotations, labels, filename = batch
        output = self(images)
        feas, c_fea, seg = output
        
        B, C, H, W = c_fea.shape

        edge_map = self.edge_conv2d(annotations)
        for b in range(B):
            vutils.save_image(edge_map[b].unsqueeze(dim=0).float() / edge_map.max(), f'/home/lib/generate_seg/output/test-{b}.png')
        
        # Training generator
        patch, valid = self.get_g_patch_valid(annotations, c_fea, edge_map)
        g_loss = self.adversarial_loss(self.discriminator(patch), valid.long())
        segment_loss = self.segment_loss(seg, annotations)
        self.log(f'train/g_loss', g_loss)
        self.log(f'train/segment_loss', segment_loss)
        g_opt.zero_grad()
        self.manual_backward(g_loss * self.g_loss_weight + segment_loss)
        g_opt.step()
        g_sch.step()

        # Training discriminator
        patch, valid = self.get_d_patch_valid(annotations, c_fea, edge_map)
        d_loss = self.adversarial_loss(self.discriminator(patch.detach()), valid.long())
        self.log(f'train/d_loss', d_loss)
        d_opt.zero_grad()
        self.manual_backward(d_loss)
        d_opt.step()
        d_sch.step()

        return {'output': output}
    
    def validation_step(self, batch, batch_idx):
        
        images, annotations, labels, filename = batch
        output = self(images)
        feas, c_fea, seg = output

        B, C, H, W = c_fea.shape
        half_patch_size = self.patch_size // 2

        edge_map = self.edge_conv2d(annotations)
        for b in range(B):
            vutils.save_image(edge_map[b].unsqueeze(dim=0).float() / edge_map.max(), f'/home/lib/generate_seg/output/test-{b}.png')

        # Validation generator
        patch, valid = self.get_g_patch_valid(annotations, c_fea, edge_map)
        g_loss = self.adversarial_loss(self.discriminator(patch), valid.long())
        segment_loss = self.segment_loss(seg, annotations)
        self.log(f'val/g_loss', g_loss)
        self.log(f'val/segment_loss', segment_loss)
        self.calculate_metric(output, batch, batch_idx, "val")

        # Validation discriminator
        patch, valid = self.get_d_patch_valid(annotations, c_fea, edge_map)
        d_loss = self.adversarial_loss(self.discriminator(patch), valid.long())
        self.log(f'val/d_loss', d_loss)

        return {'loss': None, 'output': output}

    def configure_optimizers(self):

        opt_g = torch.optim.Adam(self.generator.parameters(), lr=self.learning_rate)
        opt_d = torch.optim.AdamW(self.discriminator.parameters(), lr=0.0005, weight_decay=0.001)

        scheduler_g = torch.optim.lr_scheduler.OneCycleLR(optimizer=opt_g,
                                                          max_lr=self.learning_rate,
                                                          total_steps=self.trainer.max_steps)
        scheduler_d = LinearLR(optimizer=opt_d,
                               start_factor=0.0005,
                               end_factor=0.0,
                               total_iters=self.trainer.max_steps)

        return [opt_g, opt_d], [scheduler_g, scheduler_d]
    
    def calculate_metric(self, out, batch, batch_idx, mode):
        images, annotations, labels, filename = batch
        feas, c_fea, seg = out
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

    def log_images(self, batch, outputs):

        images, annotations, labels, filename = batch
        feas, c_fea, preds = outputs["output"]
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

# class LinearLrDecay(object):
#     def __init__(self, optimizer, start_lr, end_lr, decay_start_step, decay_end_step):

#         assert start_lr > end_lr
#         self.optimizer = optimizer
#         self.delta = (start_lr - end_lr) / (decay_end_step - decay_start_step)
#         self.decay_start_step = decay_start_step
#         self.decay_end_step = decay_end_step
#         self.start_lr = start_lr
#         self.end_lr = end_lr

#     def step(self, current_step):
#         if current_step <= self.decay_start_step:
#             lr = self.start_lr
#         elif current_step >= self.decay_end_step:
#             lr = self.end_lr
#         else:
#             lr = self.start_lr - self.delta * (current_step - self.decay_start_step)
#             for param_group in self.optimizer.param_groups:
#                 param_group['lr'] = lr
#         return lr