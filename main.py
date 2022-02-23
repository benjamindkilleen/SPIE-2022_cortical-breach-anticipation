#!/usr/bin/env python3
import logging
from pathlib import Path

import cortical_breach_detection
import hydra
import matplotlib as mpl
import pycuda.autoinit  # noqa
import pytorch_lightning as pl
from cortical_breach_detection.datasets.ctpelvic1k import CTPelvic1KDataModule
from cortical_breach_detection.models.classifier import BinaryClassifier
from omegaconf import DictConfig
from omegaconf import OmegaConf

mpl.use("agg")

log = logging.getLogger("main")


@cortical_breach_detection.register_experiment
def train(cfg):
    pl.seed_everything(cfg.seed)
    dm = CTPelvic1KDataModule(**OmegaConf.to_container(cfg.data, resolve=True))
    dm.prepare_data()
    dm.setup(stage="fit")
    model = BinaryClassifier(pos_weight=dm.train_set.pos_weight, **cfg.model)
    trainer = pl.Trainer(**cfg.trainer)
    trainer.fit(model, datamodule=dm)

    dm.setup(stage="test")
    trainer.test(model, datamodule=dm)


@cortical_breach_detection.register_experiment
def test(cfg):
    dm = CTPelvic1KDataModule(**OmegaConf.to_container(cfg.data, resolve=True))
    dm.prepare_data()
    model = BinaryClassifier.load_from_checkpoint(cfg.checkpoint, **cfg.model)
    trainer = pl.Trainer(**cfg.trainer)
    dm.setup(stage="test")
    trainer.test(model, datamodule=dm)


@hydra.main(config_path="conf", config_name="config")
def main(cfg: DictConfig) -> None:
    log.info(OmegaConf.to_yaml(cfg))
    cortical_breach_detection.run(cfg)


if __name__ == "__main__":
    main()
