from typing import Dict, Any
import os

import numpy as np
import pytorch_lightning as pl
from pytorch_lightning.utilities.types import STEP_OUTPUT
import torch
import torchvision
from PIL import Image
from pytorch_lightning.callbacks import Callback
from pytorch_lightning.utilities.rank_zero import rank_zero_only

class ImageLogger(Callback):
    """
    Log images during training or validating.
    
    TODO: Support validating.
    """
    
    def __init__(
        self,
        log_every_n_steps: int=2000,
        max_images_each_step: int=4,
        log_images_kwargs: Dict[str, Any]=None
    ) -> "ImageLogger":
        
        super().__init__()
        
        self.log_every_n_steps = log_every_n_steps
        self.max_images_each_step = max_images_each_step
        self.log_images_kwargs = log_images_kwargs or dict()

    def setup(self, trainer: pl.Trainer, pl_module: pl.LightningModule, stage: str) -> None:
        
        super().setup(trainer, pl_module, stage)
    
        # save dir
        if len(trainer.loggers) > 0:
            if trainer.loggers[0].save_dir is not None:
                save_dir = trainer.loggers[0].save_dir
            else:
                save_dir = trainer.default_root_dir
            name = trainer.loggers[0].name
            version = trainer.loggers[0].version
            version = version if isinstance(version, str) else f"version_{version}"
            self.output_dir = os.path.join(save_dir, str(name), version, "images")
        else:
            self.output_dir = os.path.join(trainer.default_root_dir, str(name), version, "images")

        os.makedirs(self.output_dir, exist_ok=True)

    @rank_zero_only
    def on_validation_batch_end(
        self, trainer: pl.Trainer, pl_module: pl.LightningModule, outputs: STEP_OUTPUT,
        batch: Any, batch_idx: int
    ) -> None:
        
        # if pl_module.global_step % self.log_every_n_steps == 0:
        #     is_train = pl_module.training
        #     if is_train:
        #         pl_module.freeze()

        if batch_idx != 0:
            return

        with torch.no_grad():
            # returned images should be: nchw, rgb, [0, 1]
            images: Dict[str, torch.Tensor] = pl_module.log_images(batch, outputs)
        
        # save images
        for image_key in images:
            
            image = images[image_key].detach().cpu()

            grid = torchvision.utils.make_grid(image, nrow=4)
            grid = grid.transpose(0, 1).transpose(1, 2).squeeze(-1).numpy()
            grid = (grid * 255).clip(0, 255).astype(np.uint8)
            
            filename = "step-{:06}_{}_e-{:06}_b-{:06}.png".format(
                pl_module.global_step, image_key, pl_module.current_epoch, batch_idx
            )
            path = os.path.join(self.output_dir, filename)
            
            Image.fromarray(grid).save(path)
        
        # if is_train:
        #     pl_module.unfreeze()
