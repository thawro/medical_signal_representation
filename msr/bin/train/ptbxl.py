import logging

import hydra
import omegaconf

log = logging.getLogger(__name__)
log.setLevel(logging.INFO)

from msr.utils import logger_name_to_str, logger_project_to_str, print_config_tree


@hydra.main(version_base=None, config_path="../../../configs/train_model", config_name="train")
def main(cfg: omegaconf.DictConfig):

    cfg.logger.project = logger_project_to_str(cfg.logger.project)
    cfg.logger.name = logger_name_to_str(cfg.logger.name)
    log.info(f"Project: {cfg.logger.project}")
    log.info(f"Run: {cfg.logger.name}")

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

    representation_type, model_name = cfg.logger.name.split("__")
    params = dict(dataset=cfg.logger.project, representation=representation_type, model=model_name)
    logger.log(params)
    results = trainer.evaluate(plotter=plotter, logger=logger)

    log.info("Evaluation finished.")
    logger.finish()


if __name__ == "__main__":
    main()
