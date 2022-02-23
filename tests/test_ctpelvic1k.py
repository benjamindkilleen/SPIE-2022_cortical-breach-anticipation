#!/usr/bin/env python3
#!/usr/bin/env python3
import hydra
import pytorch_lightning as pl
from omegaconf import DictConfig

import cortical_breach_detection
from cortical_breach_detection.datasets.ctpelvic1k import CTPelvic1K


@cortical_breach_detection.register_experiment("make_dataset")
def test_make_dateset(cfg):
    pl.seed_everything(cfg.seed)

    dataset = CTPelvic1K(**cfg.dataset)


@hydra.main(config_path="../conf", config_name="config")
def main(cfg: DictConfig) -> None:
    cortical_breach_detection.run(cfg)


def test_main():
    assert True, "Well, this is embarassing."


if __name__ == "__main__":
    main()
