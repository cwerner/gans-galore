from pathlib import Path

import hydra
import pytorch_lightning as pl
from loguru import logger as log
from omegaconf import DictConfig, OmegaConf
from pl_bolts.datamodules import FashionMNISTDataModule, MNISTDataModule

# copied from pytorch lightning models
from models.basic_gan import GAN

from models.callbacks import WandbGenerativeModelImageSampler

from pytorch_lightning.loggers import WandbLogger

import logging
logging.getLogger("lightning").setLevel(logging.INFO)
logging.getLogger("wandb").setLevel(logging.ERROR)

DATAPATH = Path().home() / ".torch" / "datasets"


class UndefinedDataset(BaseException):
    pass


@hydra.main(config_path="conf", config_name="config.yaml")
def main(cfg: DictConfig) -> None:

    log.info("Config")
    log.info(OmegaConf.to_yaml(cfg))

    dm = hydra.utils.instantiate(cfg.dataset, DATAPATH)
    model = GAN(*dm.size(), hparams=cfg)

    wandb_logger = WandbLogger(project=f'gans-galore-{dm.name}', anonymous=True, tags=['demo'])

    # fast_dev_run=True, 
    trainer = pl.Trainer(progress_bar_refresh_rate=20, gpus=1, 
                         callbacks=[WandbGenerativeModelImageSampler(64, fixed=True, annotate=True)],
                         logger=wandb_logger, max_epochs=10)
    trainer.fit(model, dm)


#
# self.logger.experiment.log({"val_input_image":[wandb.Image(val_batch[0][0].cpu(), caption="val_input_image")]})

if __name__ == "__main__":
    main()
