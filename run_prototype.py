import random
import numpy as np
from tqdm import tqdm
import cv2

import torch
from torchvision import transforms

from simple_seg.env.env_handler import EnvHandler
from simple_seg.config2.config_handler import ConfigHandler
from simple_seg.config2.config_base import CONFIG
from simple_seg.metric.metric_handler import MetricHandler
from simple_seg.log.log_handler2 import LogHandler
from simple_seg.model.load_model import get_model
from simple_seg.dataset.load_dataset2 import get_dataset
from simple_seg.util import type_cast

class Trainer():

    def __init__(self, project_name, model_name, dataset_name) -> None:

        self.init_env_handler()
        self.init_model(model_name)
        self.init_dataset_handler(dataset_name)
        self.init_metric_handler()
        self.init_optmizer_handler()
        self.init_scheduler_handler()
        self.init_loss_handler()
        self.init_log_handler(project_name)
        self.init_config_handler()
    
    def init_env_handler(self):
        self.env_handler = EnvHandler()
    
    def init_model(self, model_name):
        self.model = get_model(model_name)()
        self.model = self.env_handler.init_DDP(self.model)

    def init_dataset_handler(self, dataset_name):
        dataset_handler = get_dataset(dataset_name)
        self.train_dataset_handler = dataset_handler(mode='train', num_workers=1)
        self.eval_dataset_handler = dataset_handler(mode='eval', num_workers=1)

    def init_tensor(self, size):
        # 张量构造器
        def gen_tensor():
            return torch.zeros(size=(size,))
        return gen_tensor
    
    def gen_from_dict(self, src_dict):
        # dict构造器
        dest_dict = dict.fromkeys(src_dict.keys())
        for key in src_dict.keys():
            dest_dict[key] = src_dict[key]()
            if CONFIG.TRAIN.CUDA:
                dest_dict[key] = dest_dict[key].to(self.env_handler.device)
        
        return dest_dict

    def init_metric(self):
        self.metric_handler = {}
        self.metric_dict = {'f_score': self.init_tensor(size=CONFIG.TRAIN.NUM_CLASSES),
                            'iou': self.init_tensor(size=CONFIG.TRAIN.NUM_CLASSES),
                            'recall': self.init_tensor(size=CONFIG.TRAIN.NUM_CLASSES),
                            'precision': self.init_tensor(size=CONFIG.TRAIN.NUM_CLASSES)}

    def init_metric_handler(self):
        self.metric_handler = {}
        self.metric_dict = {}
        
        self.init_metric()

        self.train_batch_metric = self.gen_from_dict(self.metric_dict)
        self.train_metric       = self.gen_from_dict(self.metric_dict)
        self.eval_batch_metric  = self.gen_from_dict(self.metric_dict)
        self.eval_metric        = self.gen_from_dict(self.metric_dict)
        
    def init_optmizer_handler(self):
        self.optimizer_handler = None

    def init_scheduler_handler(self):
        self.scheduler_handler = None
    
    def init_loss(self):
        self.loss_handler = {}
        self.loss_dict = {'seg_loss': self.init_tensor(size=1)}
    
    def init_loss_handler(self):
        self.loss_handler = {}
        self.loss_dict = {}

        self.init_loss()
        
        self.train_batch_loss       = self.gen_from_dict(self.loss_dict)
        self.train_batch_total_loss = 0
        self.train_loss             = self.gen_from_dict(self.loss_dict)
        self.train_total_loss       = 0

        self.eval_batch_loss        = self.gen_from_dict(self.loss_dict)
        self.eval_batch_total_loss  = 0
        self.eval_loss              = self.gen_from_dict(self.loss_dict)
        self.eval_total_loss        = 0
    
    def init_log_handler(self, project_name):
        if self.env_handler.local_rank == 0:
            self.log_handler = LogHandler(project_name, self.model)
            return
        self.log_handler = None
    
    def init_config_handler(self):
        self.config_handler = ConfigHandler()
        if self.env_handler.local_rank == 0:
            self.config_handler.show_config(width=100)

    def data_from_btach(self, batch):
        images, annotations = batch
        labels = type_cast.index_to_onehot(annotations=annotations, N=CONFIG.TRAIN.NUM_CLASSES)

        # Load on GPU
        if CONFIG.TRAIN.CUDA:
            images      = images.cuda(self.env_handler.local_rank)
            annotations = annotations.cuda(self.env_handler.local_rank)
            labels      = labels.cuda(self.env_handler.local_rank)
        
        self.batch = [images, annotations, labels]

    def train_forward(self):
        images, annotations, labels = self.batch
        self.train_outputs = self.model(images)

    def train_backward(self):
        self.optimizer_handler.zero_grad()  # Zero gradient
        self.train_batch_total_loss.backward()
        self.optimizer_handler.step()       # Update gradient
    
    def eval_forward(self):
        images, annotations, labels = self.batch
        self.eval_outputs = self.model(images)
    
    def calculate_loss(self, outputs, loss):
        images, annotations, labels = self.batch

        seg_loss = self.loss_handler['seg_loss'](outputs, annotations)
        loss['seg_loss'] = self.env_handler.DPP_reduce(seg_loss)

        total_loss = seg_loss

        return total_loss
    
    def calculate_metric(self, outputs, metric):
        images, annotations, labels = self.batch

        iou = self.metric_handler['iou'](outputs.argmax(dim=1), annotations)
        recall = self.metric_handler['recall'](outputs.argmax(dim=1), annotations)
        precision = self.metric_handler['precision'](outputs.argmax(dim=1), annotations)
        f_score = self.metric_handler['f_score'](outputs.argmax(dim=1), annotations)

        metric['iou']       = self.env_handler.DPP_reduce(iou)
        metric['recall']    = self.env_handler.DPP_reduce(recall)
        metric['precision'] = self.env_handler.DPP_reduce(precision)
        metric['f_score']   = self.env_handler.DPP_reduce(f_score)

    def log_loss(self, cur_epoch, loss_name, loss_value):
        self.log_handler.log_epoch(epoch=cur_epoch,
                                   key=loss_name,
                                   value=loss_value)
        # Output
        print('%-8s %40s: %-20.4e' % (f'[{cur_epoch}]', loss_name, loss_value))

    def log_metric(self, cur_epoch, metric_name, metric_value):
        self.log_handler.log_epoch(epoch=cur_epoch,
                                   key=metric_name,
                                   value=metric_value)
        # Output
        print('%-8s %40s: %-20.4e' % (f'[{cur_epoch}]', metric_name, metric_value))
    
    def log_scheduler(self):
        self.log_handler.log_epoch(epoch=self.cur_epoch,
                                   key=f'train_scheduler',
                                   value=self.optimizer_handler.state_dict()['param_groups'][0]['lr'])

    def eval_visualize(self):
        
        image_name = ''
        images, annotations, labels = self.batch

        batch_choice = random.choice(list(range(images.shape[0])))

        image = self.train_dataset_handler.denormalize(images[batch_choice])

        trans = transforms.ToPILImage()
        vis = cv2.cvtColor(np.array(trans(image)), cv2.COLOR_RGB2BGR)
        
        self.log_handler.vis_epoch(epoch=self.cur_epoch,
                                   image_name=image_name,
                                   image=vis)

    def train_iteration(self, cur_iteration, batch):

        self.cur_iteration = cur_iteration

        self.data_from_btach(batch) # Get training data

        self.train_forward()    # Forward
        self.train_batch_total_loss = self.calculate_loss(self.train_outputs, self.train_batch_loss)
        self.train_backward()   # Backward

    def end_train_iteration(self):

        # 累计各个iteration的loss
        for key in self.train_batch_loss.keys():
            self.train_loss[key] += self.train_batch_loss[key].clone()
        self.train_total_loss += self.train_batch_total_loss.clone()

        # 显示当前iteration的loss
        if self.env_handler.local_rank == 0:
            postfix = {
                'loss'  : self.train_batch_total_loss.item(), 
                'lr'    : self.optimizer_handler.state_dict()['param_groups'][0]['lr']
            }
            self.pbar.set_postfix(**postfix)
            self.pbar.update(1)

    def train_epoch(self, cur_epoch):

        self.cur_epoch = cur_epoch

        # 初始化
        self.train_total_loss = 0
        for key in self.train_loss.keys():
            self.train_loss[key] = 0
        for key in self.train_metric.keys():
            self.train_metric[key] = 0

        # Set distribute sampler at current epoch
        self.train_dataset_handler.sampler.set_epoch(cur_epoch)

        # Start train
        self.model.train()

        # Init train progress
        if self.env_handler.local_rank == 0:
            self.pbar = tqdm(
                total=len(self.train_dataset_handler.data_loader),
                desc=f'Train Epoch {cur_epoch} / {CONFIG.TRAIN.END_EPOCH}',
                postfix=dict,
                mininterval=0.3)

        # Run iteration
        for iteration, batch in enumerate(self.train_dataset_handler.data_loader):

            self.train_iteration(cur_iteration=iteration, batch=batch)
            self.end_train_iteration()
        
        # End train progress
        if self.env_handler.local_rank == 0:
            self.pbar.close()


    def eval_iteration(self, cur_iteration, batch, visible):
        
        self.cur_iteration = cur_iteration

        self.data_from_btach(batch) # Get eval data
        
        with torch.no_grad():
            self.eval_forward()
            self.eval_batch_total_loss = self.calculate_loss(self.eval_outputs, self.eval_batch_loss)
            self.calculate_metric(self.eval_outputs, self.eval_batch_metric)
            
            if visible and self.env_handler.local_rank == 0:
                self.eval_visualize()

    def end_eval_iteration(self):

        # 累计各个iteration的loss
        for key in self.eval_batch_loss.keys():
            self.eval_loss[key] += self.eval_batch_loss[key].item()
        self.eval_total_loss += self.eval_batch_total_loss.item()

        # 累计各个iteration的metric
        for key in self.eval_batch_metric.keys():
            self.eval_metric[key] += self.eval_batch_metric[key].item()

        # 显示当前iteration的loss
        if self.env_handler.local_rank == 0:
            postfix = {
                'loss'  : self.eval_batch_total_loss.item(), 
                'lr'    : self.optimizer_handler.state_dict()['param_groups'][0]['lr'],
            }
            for key in self.eval_batch_metric.keys():
                postfix[key] = self.eval_batch_metric[key].tolist()
            self.pbar.set_postfix(**postfix)
            self.pbar.update(1)

    def eval_epoch(self, cur_epoch):

        self.cur_epoch = cur_epoch

        # 初始化
        self.eval_total_loss = 0
        for key in self.eval_loss.keys():
            self.eval_loss[key] = 0
        for key in self.eval_metric.keys():
            self.eval_metric[key] = 0

        self.model.eval()

        # Init eval progress
        if self.env_handler.local_rank == 0:
            self.pbar = tqdm(
                total=len(self.eval_dataset_handler.data_loader),
                desc=f'Validation Epoch { cur_epoch } / { CONFIG.TRAIN.END_EPOCH }',
                postfix=dict,
                mininterval=0.3
            )

        for iteration, batch in enumerate(self.eval_dataset_handler.data_loader):
            
            self.eval_iteration(cur_iteration=iteration, 
                                batch=batch,
                                visible=(iteration == cur_epoch % len(self.eval_dataset_handler.data_loader)))
            self.end_eval_iteration()

        # End eval progress
        if self.env_handler.local_rank == 0:
            self.pbar.close()
        
    def run(self):

        torch.distributed.barrier()
        
        start_epoch = CONFIG.TRAIN.START_EPOCH
        end_epoch   = CONFIG.TRAIN.END_EPOCH

        for cur_epoch in range(start_epoch, end_epoch):

            #---------------------------------------#
            #  Train model
            #---------------------------------------#
            self.train_epoch(cur_epoch=cur_epoch)

            #--------------------------------------------------------
            #   Weight decay
            #--------------------------------------------------------
            if self.env_handler.local_rank == 0:
                self.log_scheduler()
            self.scheduler_handler.step()
            
            #--------------------------------------------------------
            #   Eval model
            #--------------------------------------------------------

            eval_flag = CONFIG.TRAIN.EVAL_FLAG and (cur_epoch == 0 or (cur_epoch + 1)  % CONFIG.TRAIN.EVAL_PERIOD == 0)
            
            if eval_flag:

                self.eval_epoch(cur_epoch)
                
            #--------------------------------------------------------
            #   Record model
            #--------------------------------------------------------
            if self.env_handler.local_rank == 0:
                
                #-------------------------------------------
                #   记录 loss
                #-------------------------------------------
                
                # Log train loss
                iter_num = len(self.train_dataset_handler.data_loader)
                for key in self.train_loss.keys():
                    self.log_loss(cur_epoch=cur_epoch,
                                  loss_name=f'train_{key}',
                                  loss_value=self.train_loss[key]/iter_num)

                if eval_flag:
                    
                    iter_num = len(self.eval_dataset_handler.data_loader)
                    
                    # Log validation loss
                    for key in self.eval_loss.keys():
                        self.log_loss(cur_epoch=cur_epoch,
                                      loss_name=f'eval_{key}',
                                      loss_value=self.eval_loss[key]/iter_num)
                    
                    # Log validation metric
                    for key in self.eval_metric.keys():
                        self.log_metric(cur_epoch=cur_epoch,
                                        metric_name=f'eval_{key}',
                                        metric_value=self.eval_metric[key]/iter_num)

                #-----------------------------------------------#
                #   根据设置的保存时间间隔保存模型state
                #-----------------------------------------------#
                if (cur_epoch + 1) % CONFIG.TRAIN.SAVE_PERIOD == 0 or cur_epoch + 1 == end_epoch:

                    save_name = 'epoch%03d-loss%.3f.pth' % ((cur_epoch + 1), self.train_total_loss.item()/iter_num)
                    self.log_handler.save_model(save_name, self.model.state_dict())

                #-----------------------------------------------#
                #   保存上一次训练模型state
                #-----------------------------------------------#  
                self.log_handler.save_model('last_epoch.pth', self.model.state_dict())        
        

import cv2
import numpy as np
import sympy as sp
from scipy.stats import ortho_group
from sklearn.manifold import MDS
from sklearn.metrics.pairwise import manhattan_distances, euclidean_distances

import torch
from torch.optim import SGD
from torch.optim.lr_scheduler import OneCycleLR
from torch.nn import CrossEntropyLoss
from torch.nn import functional as F

from simple_seg.config2.config_base import CONFIG
from simple_seg.train.trainer_iter import TrainerIter
from simple_seg.util import type_cast

project_name    = 'TunnelDefect'
model_name      = 'PrototypeSegFormer'
dataset_name    = 'TunnelDefectV5Patch2'

def update_config():

    CONFIG.TRAIN.update({
        'NUM_CLASSES': 3,
        'NAME_CLASSES': ['background', 'crack', 'tile_peeling'],
        
        'CUDA': True,
        'DISTRIBUTED': True,
        'GPU_NUM': 1,
        'SYNC_BN': True,
        'FP16': False,
        'USE_WANDB': True,

        'START_EPOCH' : 0,
        'END_EPOCH'   : 100,
        'CHECKPOINT'  : '',
        'INPUT_SHAPE' : [512, 512],
        'BATCH_SIZE_PER_GPU' : 32,
        'SAVE_PERIOD' : 10,
        'SAVE_DIR'    : '/home/output/prototype_former',
        'EVAL_FLAG'   : True,
        'EVAL_PERIOD' : 1,
        'LR'          : 1e-3,
    })

class MyTrainer(TrainerIter):
    
    def __init__(self, project_name, model_name, dataset_name) -> None:
        super().__init__(project_name, model_name, dataset_name)
        self.prototypes = None

    def init_optmizer_handler(self):
        from torch.optim import Adam
        self.optimizer_handler = Adam(self.model.parameters(),
                                      lr=CONFIG.TRAIN.LR)
    
    def init_scheduler_handler(self):
        self.scheduler_handler = OneCycleLR(self.optimizer_handler, 
                                            max_lr=CONFIG.TRAIN.LR, 
                                            epochs=CONFIG.TRAIN.END_EPOCH,
                                            steps_per_epoch=14)
    
    def init_loss_handler(self):
        self.loss_handler = {}

        from simple_seg.loss.focal_loss import FocalLoss
        self.loss_handler['mask_loss'] = FocalLoss(torch.tensor([0.5, 0.9, 0.8]).to(self.env_handler.device)).to(self.env_handler.device)
        self.loss_handler['prototype_loss'] = FocalLoss(torch.tensor([1.0, 1.0, 1.0]).to(self.env_handler.device)).to(self.env_handler.device)

    def train_forward(self, images):
        self.train_outputs = self.model(images)
    
    def train_backward(self, loss):
        total_loss = loss['mask_loss'] + loss['prototype_loss'] * 0.1
        total_loss.backward()
    
    def eval_forward(self, images):
        self.val_outputs = self.model(images)

    def update_prototype(self, new_prototype, C_FEATURE, dim):
        if self.prototypes is None:
            mat = ortho_group.rvs(dim=C_FEATURE)[ : CONFIG.TRAIN.NUM_CLASSES, : ] * 0.1
            self.prototypes = torch.FloatTensor(mat).cuda(self.env_handler.device)
        if torch.sum(new_prototype) > 1e-4:
            # normalized_tensor_1 = new_prototype / new_prototype.norm(dim=-1, keepdim=True)
            # normalized_tensor_2 = self.prototypes[dim] / self.prototypes[dim].norm(dim=-1, keepdim=True)
            # similarity = ((normalized_tensor_1 * normalized_tensor_2).sum(dim=-1) + 1) / 2
            similarity = 0.2
            self.prototypes[dim] = similarity * new_prototype  + (1 - similarity) * self.prototypes[dim]
    
    def get_patch_prototype(self, images, proj, x, annotations, labels, patch_size, dim):
        patch_stride = patch_size // 2
        patch_padding = patch_size // 4
        
        with torch.no_grad():
            cam = x * labels + labels
            cam = torch.softmax(torch.max_pool2d(cam,
                                                 kernel_size=patch_size,
                                                 stride=patch_stride,
                                                 padding=patch_padding).detach(), dim=1)
            cam_c = cam[:, dim, :, :]
            # Min-max normalized
            B_CAM, H_CAM, W_CAM = cam_c.shape
            cam_max = torch.max(cam_c.view(B_CAM, -1), dim=-1)[0].view(B_CAM, 1, 1)
            cam_min = torch.min(cam_c.view(B_CAM, -1), dim=-1)[0].view(B_CAM, 1, 1)
            cam_c = (cam_c - cam_min - 1e-5) / (cam_max - cam_min + 1e-5)

            feature = proj.detach()
            C_FEATURE = feature.shape[1]
            feature = feature.permute(0, 2, 3, 1).reshape(-1, C_FEATURE)

            # Find top K
            top_values, top_indices = torch.topk(input=cam_c.reshape(-1),
                                                 k=patch_size // 4,
                                                 dim=-1)

            prototype_c = torch.zeros(C_FEATURE).cuda(self.env_handler.device)
            
            # Calculate prototype
            top_fea = feature[top_indices]
            prototype_c = torch.sum(top_values.unsqueeze(-1) * top_fea, dim=0) / (torch.sum(top_values) + 1e-5)

            self.update_prototype(prototype_c, C_FEATURE, dim)
        

    def calculate_loss(self, images, outputs, annotations, labels):
        x, proj4, proj32 = outputs
        
        # Init Prototype
        self.get_patch_prototype(images, F.interpolate(proj32, proj4.shape[2 : ], mode='bilinear'), x, annotations, labels, 4 * 2, dim=1)
        self.get_patch_prototype(images, proj32, x, annotations, labels, 32 * 2, dim=0)
        self.get_patch_prototype(images, proj32, x, annotations, labels, 32 * 2, dim=2)
        # L2 Norm
        self.prototypes = self.env_handler.DPP_reduce(self.prototypes)
        orth_prototypes = []
        for i in range(self.prototypes.shape[0]):
            orth_prototypes.append(sp.Matrix(self.prototypes[i].detach().cpu().numpy()))
        orth_prototypes = sp.GramSchmidt(orth_prototypes, orthonormal=True)
        orth_prototypes = np.array(orth_prototypes).astype(np.float32).squeeze()
        self.prototypes = torch.tensor(orth_prototypes).to(self.env_handler.device)
        self.prototypes = F.normalize(self.prototypes, dim=-1)
        if self.env_handler.local_rank == 0:
            for i in range(CONFIG.TRAIN.NUM_CLASSES - 1):
                for j in range(i + 1, CONFIG.TRAIN.NUM_CLASSES):
                    print(f'dim{i}-{j}', torch.sum(self.prototypes[i] * self.prototypes[j]))

        # Pixel feature [BxHxW, C_FEATURE]
        B_F, C_F, H_F, W_F = proj32.shape
        proj = proj32.permute(0, 2, 3, 1).reshape(B_F * H_F * W_F, C_F)
        proj = F.normalize(proj, dim=-1)
        
        # Pixel prototype [BxHxW, C_FEATURE]
        label_patch = torch.max_pool2d(labels,
                                       kernel_size=64,
                                       stride=32,
                                       padding=16).argmax(dim=1).detach()
        
        positives =  self.prototypes[label_patch.reshape(-1)]

        # All prototype
        A1 = torch.exp(torch.sum(proj * positives, dim=-1) / 0.1)
        A2 = torch.sum(torch.exp(torch.matmul(proj, self.prototypes.transpose(0, 1)) / 0.1), dim=-1)
        loss_nce32 = torch.mean(-1 * torch.log(A1 / A2))

        return {
            'mask_loss': self.loss_handler['mask_loss'](x, annotations),
            'prototype_loss': loss_nce32
        }
    
    def aggregate_loss(self, total_loss, loss):
        mask_loss       = loss['mask_loss']
        prototype_loss  = loss['prototype_loss']
        mask_loss       = self.env_handler.DPP_reduce(mask_loss)
        prototype_loss  = self.env_handler.DPP_reduce(prototype_loss)
        total_loss.append({'mask_loss'      : mask_loss.item() if torch.is_tensor(mask_loss) else mask_loss,
                           'prototype_loss' : prototype_loss.item() if torch.is_tensor(prototype_loss) else prototype_loss})
    
    def log_loss(self, cur_epoch, loss_name, total_loss):
        for sub_loss_name in total_loss[0].keys():
            loss_list = [l[sub_loss_name] for l in total_loss]
            avg_loss = np.average(loss_list)
            self.log_handler.log_epoch(epoch=cur_epoch,
                                       key=f'{loss_name}_{sub_loss_name}',
                                       value=avg_loss)
    
    def calculate_metric(self, metrics, outputs, annotations, labels):
        x, proj4, proj32 = outputs

        # f_scores = self.metric_handler.get_f_score(x, labels)
        ious, recalls, precisions = self.metric_handler.get_moiu(x.argmax(dim=1), annotations)
        beta = 2
        f_scores = (1 + beta**2) * (precisions * recalls) / (beta**2 * precisions + recalls + 1e-6)
        
        metrics['f_score']   += self.env_handler.DPP_reduce(f_scores)
        metrics['iou']       += self.env_handler.DPP_reduce(ious)
        metrics['recall']    += self.env_handler.DPP_reduce(recalls)
        metrics['precision'] += self.env_handler.DPP_reduce(precisions)
        
    
    def contrast_visible(self, fea, size):
        
        B, C, H, W = fea.shape
        fea = fea[0].permute(1, 2, 0).reshape(-1, C)
        dist_manhattan = manhattan_distances(fea.cpu())
        mds = MDS(n_components=1, dissimilarity='precomputed', random_state=0, normalized_stress='auto')
        # Get the embeddings
        trans_fea = mds.fit_transform(dist_manhattan)
        trans_fea = torch.tensor(trans_fea).reshape(H, W, 1).numpy()
        trans_fea = cv2.normalize(trans_fea, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        # trans_fea = cv2.cvtColor(trans_fea, cv2.COLOR_RGB2GRAY)
        trans_fea = cv2.applyColorMap(trans_fea, cv2.COLORMAP_MAGMA)
        trans_fea = cv2.resize(trans_fea, size)
        return trans_fea

    def visible(self, images, outputs, annotations):
        x, proj4, proj32 = outputs

        image_bgr = images[0].cpu().numpy().transpose(1, 2, 0) * 255
        pred_rgb = type_cast.index_to_color(x[0].argmax(dim=0))
        pred_bgr = cv2.cvtColor(pred_rgb.permute(1, 2, 0).cpu().numpy(), cv2.COLOR_RGB2BGR)
        label_rgb = type_cast.index_to_color(annotations[0])
        label_bgr = cv2.cvtColor(label_rgb.permute(1, 2, 0).cpu().numpy(), cv2.COLOR_RGB2BGR)
        
        trans_proj4 = self.contrast_visible(F.interpolate(proj4, scale_factor=0.25, mode='bilinear'), image_bgr.shape[ : 2])
        trans_proj32 = self.contrast_visible(proj32, image_bgr.shape[ : 2])
        
        image = np.hstack([image_bgr, pred_bgr, label_bgr, trans_proj4, trans_proj32])

        self.log_handler.vis_epoch(epoch=self.cur_epoch,
                                   image_name='result',
                                   image=image)

if __name__ == '__main__':

    update_config()

    trainer = MyTrainer(project_name=project_name,
                        model_name=model_name,
                        dataset_name=dataset_name)
    trainer.run()
        
