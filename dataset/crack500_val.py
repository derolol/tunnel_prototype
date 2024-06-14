import os
import random
from typing import Sequence, Dict, Union

import cv2
import numpy as np
from PIL import Image

import torch.utils.data as data
from torchvision.transforms import transforms

class Crack500Dataset(data.Dataset):
    
    def __init__(
        self,
        file_list,
        image_root,
        annotation_root,
        image_format,
        annotation_format,
        out_size,
        apply_transform,
        train=True
    ) -> "Crack500Dataset":
        
        super().__init__()

        self.file_list = self.read_file_list(file_list)
        self.image_root = image_root
        self.annotation_root = annotation_root
        self.image_format = image_format
        self.annotation_format = annotation_format
        self.out_size = (out_size, out_size)
        self.apply_transform = apply_transform
        self.train = train

    def read_file_list(self, file_list):
        paths = []
        with open(file_list, 'r') as f:
            for path in f.readlines():
                paths.append(path.strip())
        return paths
    
    def __getitem__(self, index: int) -> Dict[str, Union[np.ndarray, str]]:
        image_name = self.file_list[index]
        
        image_path = os.path.join(self.image_root, f'{image_name}.{self.image_format}')
        annotation_path = os.path.join(self.annotation_root, f'{image_name}.{self.annotation_format}')
        
        image_pil = Image.open(image_path).convert('RGB')
        annotation_pil = Image.open(annotation_path).convert('L')

        image = transforms.F.to_tensor(image_pil).float()
        annotation = (transforms.F.pil_to_tensor(annotation_pil) / 255).long()

        if self.apply_transform:

            # Crop
            i, j, h, w = transforms.RandomCrop.get_params(img=image, output_size=self.out_size)
            image = transforms.F.crop(image, top=i, left=j, height=h, width=w)
            annotation = transforms.F.crop(annotation, top=i, left=j, height=h, width=w)

            # brightness
            brightness = random.random()
            image = transforms.F.adjust_brightness(image, brightness)

            # contrast
            contrast = random.random()
            image = transforms.F.adjust_contrast(image, contrast)

        if not self.apply_transform:

            if self.train:
                image = transforms.F.center_crop(image, output_size=self.out_size)
                annotation = transforms.F.center_crop(annotation, output_size=self.out_size)
            else:
                image = transforms.F.resize(image, 
                                            size=self.out_size,
                                            interpolation=transforms.InterpolationMode.BILINEAR)
                annotation = transforms.F.resize(annotation, 
                                                 size=self.out_size,
                                                 interpolation=transforms.InterpolationMode.NEAREST)

        return image, annotation, image_name

    def __len__(self) -> int:
        return len(self.file_list)
