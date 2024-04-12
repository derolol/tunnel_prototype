from typing import (Optional, Union, Any, Mapping)
from omegaconf import OmegaConf

import torch
from torch import nn, Tensor
from torch.optim import AdamW
from pytorch_lightning import Trainer, LightningModule

from util.common import instantiate_from_config, load_state_dict


class ModelModule(LightningModule):

    def __init__(self, learning_rate, *args: torch.Any, **kwargs: torch.Any) -> None:
        super().__init__(*args, **kwargs)
        self.learning_rate = learning_rate
        self.net = nn.Conv2d(in_channels=3, out_channels=3, kernel_size=3, stride=1, padding=1)
        self.loss = nn.CrossEntropyLoss()
    
    def training_step(self, batch, batch_idx) -> Optional[Union[Tensor, Mapping[str, Any]]]:
        image, annotations, label, image_name = batch
        output = self.net(image)
        return self.loss(output, annotations)
    
    def test_step(self, batch, batch_idx) -> Optional[Union[Tensor, Mapping[str, Any]]]:
        image, annotations, label, image_name = batch
        output = self.net(image)
        return self.loss(output, annotations)

    def validation_step(self, batch, batch_idx) -> Optional[Union[Tensor, Mapping[str, Any]]]:
        image, annotations, label, image_name = batch
        output = self.net(image)
        return self.loss(output, annotations)
    

    def configure_optimizers(self):
        optmizer = AdamW(self.parameters(), self.learning_rate)
        return optmizer


def main() -> None:
    train_config = OmegaConf.load("config/test_train.yaml")

    model = ModelModule(learning_rate=1e-3)
    
    data_module = instantiate_from_config(train_config.data)

    trainer = Trainer(**train_config.lightning.trainer)
    trainer.fit(model, datamodule=data_module)

if __name__ == "__main__":
    main()
