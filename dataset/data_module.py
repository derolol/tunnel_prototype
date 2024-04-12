import random
from typing import Any, Tuple, Mapping
from omegaconf import OmegaConf

import torch
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import transforms

from pytorch_lightning.utilities.types import EVAL_DATALOADERS, TRAIN_DATALOADERS
from pytorch_lightning import LightningDataModule

from util.common import instantiate_from_config


class DataModule(LightningDataModule):
    
    def __init__(
        self,
        num_classes,
        train_config: str,
        val_config: str=None
    ) -> "DataModule":
        
        super().__init__()
        self.num_classes = num_classes
        self.train_config = OmegaConf.load(train_config)
        self.val_config = OmegaConf.load(val_config) if val_config else None

    def load_dataset(self, config: Mapping[str, Any]) -> Dataset:
        return instantiate_from_config(config["dataset"])

    def setup(self, stage: str) -> None:
        if stage == "fit":
            self.train_dataset = self.load_dataset(self.train_config)
            if self.val_config:
                self.val_dataset = self.load_dataset(self.val_config)
        else:
            raise NotImplementedError(stage)

    def train_dataloader(self) -> TRAIN_DATALOADERS:
        return DataLoader(
            dataset=self.train_dataset, **self.train_config["data_loader"]
        )

    def val_dataloader(self) -> EVAL_DATALOADERS:
        if self.val_dataset is None:
            return None
        return DataLoader(
            dataset=self.val_dataset, **self.val_config["data_loader"]
        )

    def on_after_batch_transfer(self, batch: Any, dataloader_idx: int) -> Any:
        N = self.num_classes
        image, annotation, image_name = batch
        
        # Flip
        if random.choice([True, False]):
            image = transforms.F.hflip(image)
            annotation = transforms.F.hflip(annotation)
        if random.choice([True, False]):
            image = transforms.F.vflip(image)
            annotation = transforms.F.vflip(annotation)

        # # Rotation
        # angle = transforms.RandomRotation.get_params(degrees=[-90, 90])
        # image = transforms.F.rotate(image, angle=angle,
        #                             interpolation=transforms.InterpolationMode.BILINEAR, expand=True)
        # annotation = transforms.F.rotate(annotation, angle=angle,
        #                                  interpolation=transforms.InterpolationMode.NEAREST, expand=True)

        # Normalize and squeeze
        image = transforms.F.normalize(image, mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        annotation = annotation.squeeze(dim=1)

        B, H, W = annotation.shape
        annotation_flatten = annotation.view(-1)
        ones = torch.eye(N).to(image.device)
        ones = ones.index_select(dim=0, index=annotation_flatten)
        label = ones.view(B, H, W, N)
        label = label.transpose(2, 3).transpose(1, 2)

        return image, annotation, label, image_name