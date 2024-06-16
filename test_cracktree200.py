from argparse import ArgumentParser
from omegaconf import OmegaConf

import os
import csv
import time
import cv2
import einops
from tqdm import tqdm
import numpy as np
from skimage.morphology import skeletonize

import torch
import pytorch_lightning as pl
from pytorch_lightning import seed_everything

from dataset.data_module import DataModule
from util.common import instantiate_from_config, load_state_dict

class Validation():

    log_save_dir = 'val_cracktree200'

    model_meta = {
        'segformer': {
            'config': '/home/lib/generate_seg/config/train_crack500_segformer.yaml',
            'model_path': '/home/lib/generate_seg/train_crack500_viz/crack500_segformer/without_pretrained/checkpoints/step=18000.ckpt',
            'log_name': 'segformer',
        },
        'hrnet': {
            'config': '/home/lib/generate_seg/config/train_crack500_hrnet.yaml',
            'model_path': '/home/lib/generate_seg/train_crack500_viz/crack500_hrnet/version_0/checkpoints/step=20000.ckpt',
            'log_name': 'hrnet',
        },
        'deeplabv3plus': {
            'config': '/home/lib/generate_seg/config/train_crack500_deeplabv3plus.yaml',
            'model_path': '/home/lib/generate_seg/train_crack500_viz/crack500_deeplabv3plus/version_0/checkpoints/step=18000.ckpt',
            'log_name': 'deeplabv3plus',
        },
        'bisenet': {
            'config': '/home/lib/generate_seg/config/train_crack500_bisenet.yaml',
            'model_path': '/home/lib/generate_seg/train_crack500_viz/crack500_bisenet/version_1/checkpoints/step=20000.ckpt',
            'log_name': 'bisenet',
        },
        'unet': {
            'config': '/home/lib/generate_seg/config/train_crack500_unet.yaml',
            'model_path': '/home/lib/generate_seg/train_crack500_viz/crack500_unet/version_0/checkpoints/step=16000.ckpt',
            'log_name': 'unet',
        },
        'swin_unet': {
            'config': '/home/lib/generate_seg/config/train_crack500_swin_unet.yaml',
            'model_path': '/home/lib/generate_seg/train_crack500_viz/crack500_swin_unet/version_0/checkpoints/step=20000.ckpt',
            'log_name': 'swin_unet',
        },

        'crack500_avqseg_default': {
            'config': '/home/lib/generate_seg/config/train_crack500_avqseg_default.yaml',
            'model_path': '/home/lib/generate_seg/train_crack500_viz/crack500_avqseg_default/version_3/checkpoints/step=20000.ckpt',
            'log_name': 'crack500_avqseg_default',
        },
        'crack500_avqseg_soft': {
            'config': '/home/lib/generate_seg/config/train_crack500_avqseg_softmax.yaml',
            'model_path': '/home/lib/generate_seg/train_crack500_viz/crack500_avqseg_soft/version_3/checkpoints/step=20000.ckpt',
            'log_name': 'crack500_avqseg_soft',
        },
        'crack500_avqseg_quant_fs': {
            'config': '/home/lib/generate_seg/config/train_crack500_avqseg_quant_fs.yaml',
            'model_path': '/home/lib/generate_seg/train_crack500_viz/crack500_avqseg_quant_fs/version_0/checkpoints/step=20000.ckpt',
            'log_name': 'crack500_avqseg_quant_fs',
        },
        'crack500_avqseg_refine_edge': {
            'config': '/home/lib/generate_seg/config/train_crack500_avqseg_refine_edge.yaml',
            'model_path': '/home/lib/generate_seg/train_crack500_viz/crack500_avqseg_refine_edge/version_0/checkpoints/step=20000.ckpt',
            'log_name': 'crack500_avqseg_refine_edge',
        },
        'crack500_avqseg_soft_quant_fs': {
            'config': '/home/lib/generate_seg/config/train_crack500_avqseg_softmax_quant_fs.yaml',
            'model_path': '/home/lib/generate_seg/train_crack500_viz/crack500_avqseg_soft_quant_fs/version_3/checkpoints/step=20000.ckpt',
            'log_name': 'crack500_avqseg_soft_quant_fs',
        },
        'crack500_avqseg_soft_quant_fs_refine_edge': {
            'config': '/home/lib/generate_seg/config/train_crack500_avqseg_softmax_quant_fs_refine_edge.yaml',
            'model_path': '/home/lib/generate_seg/train_crack500_viz/crack500_avqseg_soft_quant_fs_refine_edge/version_17/checkpoints/step=20000.ckpt',
            'log_name': 'crack500_avqseg_soft_quant_fs_refine_edge',
        },
        'crack500_avqseg_soft_refine_edge': {
            'config': '/home/lib/generate_seg/config/train_crack500_avqseg_softmax_refine_edge.yaml',
            'model_path': '/home/lib/generate_seg/train_crack500_viz/crack500_avqseg_soft_refine_edge/version_0/checkpoints/step=20000.ckpt',
            'log_name': 'crack500_avqseg_soft_refine_edge',
        },
        'crack500_avqseg_quant_fs_refine_edge': {
            'config': '/home/lib/generate_seg/config/train_crack500_avqseg_quant_fs_refine_edge.yaml',
            'model_path': '/home/lib/generate_seg/train_crack500_viz/crack500_avqseg_quant_fs_refine_edge/version_0/checkpoints/step=20000.ckpt',
            'log_name': 'crack500_avqseg_quant_fs_refine_edge',
        },
        # '': {
        #     'config': '',
        #     'model_path': '',
        #     'log_name': '',
        # }
    }

    def __init__(self, model_name):

        print('Validation model:', model_name)

        meta = self.model_meta[model_name]

        config = OmegaConf.load(meta['config'])
        model_module_ins = instantiate_from_config(config.model)
        self.model_module = model_module_ins.__class__.load_from_checkpoint(
            meta['model_path'],
            **config.model.params)
        self.model_module.eval()

        self.color_map = torch.tensor([[0, 0, 0], [225, 51, 51], [21, 72, 187], [60,179,113]]).long().to(self.model_module.device)

        # Log Path
        self.log_path = os.path.join(self.log_save_dir, self.model_meta[model_name]['log_name'])
        if not os.path.exists(self.log_path):
            os.makedirs(self.log_path)
        self.csv_name = f"test-{time.time()}"
        self.log_csv_path = os.path.join(self.log_path, f"{self.csv_name}.csv")
        os.mknod(self.log_csv_path)
        self.log_total_csv_path = os.path.join(self.log_path, f"{self.csv_name}_total.csv")
        os.mknod(self.log_total_csv_path)
        # Image path
        self.log_image_path = os.path.join(self.log_path, f"{self.csv_name}_image")
        if not os.path.exists(self.log_image_path):
            os.makedirs(self.log_image_path)

    def val(self, val_dataloader):

        precision = 0
        recall = 0
        iou = 0
        f1score = 0
        num = 0

        with open(self.log_csv_path, "w") as csvfile: 

            writer = csv.writer(csvfile)
            writer.writerow(["image_name", "precision", "recall", "iou", "f1score"])

            for batch_idx, batch in enumerate(tqdm(val_dataloader)):

                batch[0] = batch[0].to(self.model_module.device)
                batch[0] = batch[0] * 2.0 - 1.0
                batch[1] = batch[1].to(self.model_module.device)
            
                with torch.no_grad():
                    rows, preds = self.model_module.test_s_all(batch, batch_idx)
                
                # save csv
                writer.writerows(rows) # [[item1, item2, item3], [row2], [row3]]
                # save images
                for index in range(len(rows)):

                    # total
                    precision += sum([float(x) for x in rows[index][1].split('|')]) / 2
                    recall += sum([float(x) for x in rows[index][2].split('|')]) / 2
                    iou += sum([float(x) for x in rows[index][3].split('|')]) / 2
                    f1score += sum([float(x) for x in rows[index][4].split('|')]) / 2
                    num += 1

                    image_name = rows[index][0]
                    
                    image = batch[0][index]
                    image = (image + 1.) / 2. * 255
                    image = einops.rearrange(image, 'c h w -> h w c')
                    image = image.cpu().numpy().astype(np.uint8)

                    annotation = batch[1][index].squeeze(dim=0)
                    
                    # pred = preds[index].argmax(dim=0)
                    pred = preds[index]
                    FP_index = annotation - pred
                    # FP = pred.clone()
                    # FP[FP_index == -1] = 2
                    FN_index = annotation - pred
                    # FN = pred.clone()
                    # FN[FN_index == 1] = 3
                    # skeleton_index = skeletonize(pred.cpu().numpy().astype(np.uint8))
                    # skeleton = torch.tensor(skeleton_index).to(pred)

                    FPFN = pred.clone()
                    FPFN[FP_index == -1] = 2
                    FPFN[FN_index == 1] = 3

                    pred = self.color_map[pred]
                    pred = pred.cpu().numpy().astype(np.uint8)

                    # FP = self.color_map[FP]
                    # FP = FP.cpu().numpy().astype(np.uint8)

                    # FN = self.color_map[FN]
                    # FN = FN.cpu().numpy().astype(np.uint8)

                    FPFN = self.color_map[FPFN]
                    FPFN = FPFN.cpu().numpy().astype(np.uint8)

                    # skeleton = self.color_map[skeleton]
                    # skeleton = skeleton.cpu().numpy().astype(np.uint8)

                    image_viz = cv2.cvtColor(image, code=cv2.COLOR_RGB2BGR)
                    
                    mask_viz = cv2.addWeighted(image, 1.0, pred, 0.6, gamma=0.0)
                    mask_viz = cv2.cvtColor(mask_viz, code=cv2.COLOR_RGB2BGR)
                    
                    # mask_fp_viz = cv2.addWeighted(image, 0.7, FP, 0.5, gamma=0.0)
                    # mask_fp_viz = cv2.cvtColor(mask_fp_viz, code=cv2.COLOR_RGB2BGR)

                    # mask_fn_viz = cv2.addWeighted(image, 0.7, FN, 0.5, gamma=0.0)
                    # mask_fn_viz = cv2.cvtColor(mask_fn_viz, code=cv2.COLOR_RGB2BGR)

                    mask_fpfn_viz = cv2.addWeighted(image, 1.0, FPFN, 0.6, gamma=0.0)
                    mask_fpfn_viz = cv2.cvtColor(mask_fpfn_viz, code=cv2.COLOR_RGB2BGR)

                    # skeleton_viz = cv2.addWeighted(image, 1.0, skeleton, 0.8, gamma=0.0)
                    # skeleton_viz = cv2.cvtColor(skeleton_viz, code=cv2.COLOR_RGB2BGR)
                    
                    cv2.imwrite(os.path.join(self.log_image_path, f'{image_name}.png'), image_viz)
                    cv2.imwrite(os.path.join(self.log_image_path, f'{image_name}_mask.png'), mask_viz)
                    # cv2.imwrite(os.path.join(self.log_image_path, f'{image_name}_mask_fp.png'), mask_fp_viz)
                    # cv2.imwrite(os.path.join(self.log_image_path, f'{image_name}_mask_fn.png'), mask_fn_viz)
                    # cv2.imwrite(os.path.join(self.log_image_path, f'{image_name}_mask_fpfn.png'), mask_fpfn_viz)
                    # cv2.imwrite(os.path.join(self.log_image_path, f'{image_name}_skeleton.png'), skeleton_viz)
        
        with open(self.log_total_csv_path, "w") as csvfile: 

            writer = csv.writer(csvfile)
            writer.writerow(["precision", "recall", "iou", "f1score"])
            writer.writerow([precision / num,
                             recall / num,
                             iou / num,
                             f1score / num])
        
        print(precision / num,
              recall / num,
              iou / num,
              f1score / num)

if __name__ == "__main__":
    
    os.chdir(os.path.dirname(__file__))

    seed = 231
    seed_everything(seed, workers=True)
    
    # Load data
    data_module = DataModule(num_classes=2,
                             train_config='/home/lib/generate_seg/config/dataset/cracktree200.yaml',
                             val_config='/home/lib/generate_seg/config/dataset/cracktree200.yaml')
    data_module.setup("fit")
    val_dataloader = data_module.val_dataloader()

    for model_name in Validation.model_meta.keys():
        val_model = Validation(model_name=model_name)
        val_model.val(val_dataloader)