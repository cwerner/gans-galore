from pathlib import Path

import hydra
import pytorch_lightning as pl
from loguru import logger as log
from omegaconf import DictConfig
from pl_bolts.datamodules import FashionMNISTDataModule, MNISTDataModule
from pl_bolts.models.gans import GAN

DATAPATH = Path().home() / ".torch" / "datasets"


class UndefinedDataset:
    pass


@hydra.main(config_path="conf", config_name="config.yaml")
def main(cfg: DictConfig) -> None:

    if cfg.dataset == "mnist":
        dm = MNISTDataModule(DATAPATH)
    elif cfg.dataset == "fashionmnist":
        dm = FashionMNISTDataModule(DATAPATH)
    else:
        valid_datasets = ["mnist", "fashionmnist"]
        log.error(f"Dataset {cfg.dataset} not defined. Use these: {valid_datasets}")
        raise UndefinedDataset

    model = GAN(*dm.size())
    trainer = pl.Trainer(max_epochs=cfg.epochs, progress_bar_refresh_rate=20)
    trainer.fit(model, dm)


if __name__ == "__main__":
    main()
