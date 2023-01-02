import logging

import hydra
import omegaconf
from torchvision.transforms import Compose

log = logging.getLogger(__name__)
log.setLevel(logging.INFO)

from msr.utils import logger_name_to_str, model2str, print_config_tree


@hydra.main(version_base=None, config_path="../../configs/train_model", config_name="train")
def main(cfg: omegaconf.DictConfig):
    if "Module" in cfg.model._target_:  # DL models
        cfg.logger.name = f"{cfg.dataset}__{cfg.representation_type}__{cfg.model.net._target_}"
    cfg.logger.name = logger_name_to_str(cfg.logger.name)
    log.info(f"Project: {cfg.logger.project}")
    log.info(f"Run: {cfg.logger.name}")

    if cfg.logger._target_ == "msr.training.loggers.DLWandbLogger":
        config = omegaconf.OmegaConf.to_object(cfg)
        logger = hydra.utils.instantiate(cfg.logger)(config=config)
    else:
        logger = hydra.utils.instantiate(cfg.logger)
        logger.init(config=cfg)
    log.info(f"{logger} initialized")
    log.info(f"Logger set config and initialized")

    log.info("Training + evaluation started")
    print_config_tree(cfg, keys=["datamodule", "model", "callbacks", "logger", "plotter", "transforms"])
    train_transform = Compose(
        [hydra.utils.instantiate(transform["train_transform"]) for _, transform in cfg.transforms.items()]
    )
    inference_transform = Compose(
        [hydra.utils.instantiate(transform["inference_transform"]) for _, transform in cfg.transforms.items()]
    )
    datamodule = hydra.utils.instantiate(
        cfg.datamodule, train_transform=train_transform, inference_transform=inference_transform
    )
    log.info(f"{datamodule} initialized")
    datamodule.setup()
    log.info(f"Datamodule set up")

    if "Module" in cfg.model._target_:  # DL models
        net_factory = hydra.utils.instantiate(cfg.model.net)
        net_params = {"num_classes": datamodule.num_classes} if "Classifier" in cfg.model._target_ else {}
        if "MLP" in cfg.model.net._target_:
            net = net_factory(**net_params, input_size=datamodule.transformed_input_shape[0])
        elif "CNN" in cfg.model.net._target_:
            net = net_factory(**net_params, in_channels=datamodule.transformed_input_shape[0])
        model = hydra.utils.instantiate(cfg.model, net=net)
    else:
        model = hydra.utils.instantiate(cfg.model)
    log.info(f"{model} initialized")

    plotter = hydra.utils.instantiate(cfg.plotter)
    log.info(f"{plotter} initialized")

    trainer = hydra.utils.instantiate(cfg.trainer)(model=model, datamodule=datamodule)
    log.info(f"{trainer} initialized")

    log.info(f"Trainer fit started")
    trainer.fit()
    log.info(f"Trainer fit finished")

    dataset, representation_type, model_name = cfg.logger.name.split("__")
    params = dict(dataset=dataset, representation=representation_type, model=model2str[model_name])  # logger.project,
    logger.log(params)
    results = trainer.evaluate(plotter=plotter, logger=logger)

    log.info("Evaluation finished.")
    logger.finish()


if __name__ == "__main__":
    main()
