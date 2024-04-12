from argparse import ArgumentParser
from omegaconf import OmegaConf

import torch
import pytorch_lightning as pl

from util.common import instantiate_from_config, load_state_dict


def main() -> None:
    parser = ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()
    
    config = OmegaConf.load(args.config)
    pl.seed_everything(config.lightning.seed, workers=True)
    
    data_module = instantiate_from_config(config.data)
    model = instantiate_from_config(config.model)
    
    callbacks = []
    for callback_config in config.lightning.callbacks:
        callbacks.append(instantiate_from_config(callback_config))
    
    logger = instantiate_from_config(config.lightning.logger)

    trainer = pl.Trainer(logger=logger, callbacks=callbacks, **config.lightning.trainer)
    trainer.fit(model, datamodule=data_module)


if __name__ == "__main__":
    main()
