import logging

import hydra
import omegaconf

log = logging.getLogger(__name__)
log.setLevel(logging.INFO)

import wandb
from msr.utils import ROOT, print_config_tree


@hydra.main(version_base=None, config_path="../../../configs/train_model", config_name="train")
def main(cfg: omegaconf.DictConfig):
    logger = hydra.utils.instantiate(cfg.logger)
    log.info(f"{logger} initialized")
    logger.init(config=cfg)
    log.info(f"Logger set config and initialized")

    log.info("PTB-XL training + evaluation started")
    print_config_tree(cfg, keys=["datamodule", "model", "callbacks", "logger", "plotter"])

    datamodule = hydra.utils.instantiate(cfg.datamodule)
    log.info(f"{datamodule} initialized")
    datamodule.setup()
    log.info(f"Datamodule set up")

    model = hydra.utils.instantiate(cfg.model)
    log.info(f"{model} initialized")

    plotter = hydra.utils.instantiate(cfg.plotter)
    log.info(f"{plotter} initialized")

    trainer = hydra.utils.instantiate(cfg.trainer)(model=model, datamodule=datamodule)
    log.info(f"{trainer} initialized")

    trainer.fit()
    log.info(f"Trainer fit finished")

    results = trainer.evaluate(plotter=plotter, logger=logger)
    log.info("Evaluation finished.")


if __name__ == "__main__":
    main()
