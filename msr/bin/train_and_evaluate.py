import logging

import hydra
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning import LightningDataModule
from torchvision.transforms import Compose

from msr.utils import logger_name_to_str, model2str, print_config_tree

log = logging.getLogger(__name__)
log.setLevel(logging.INFO)


def parse_cfg(cfg: DictConfig):
    if "Module" in cfg.model._target_:  # DL models
        cfg.logger.name = f"{cfg.dataset}__{cfg.representation_type}__{cfg.model.net._target_}"
    cfg.logger.name = logger_name_to_str(cfg.logger.name)
    log.info(f"Project: {cfg.logger.project}")
    log.info(f"Run: {cfg.logger.name}")
    return cfg


def create_logger(cfg: DictConfig):
    if cfg.logger._target_ == "msr.training.loggers.DLWandbLogger":
        config = OmegaConf.to_object(cfg)
        logger = instantiate(cfg.logger)(config=config)
    else:
        logger = instantiate(cfg.logger)
        logger.init(config=cfg)
    log.info(f"{logger.__class__.__name__} initialized")
    log.info(f"Logger set config and initialized")
    return logger


def create_datamodule(cfg: DictConfig):
    transforms = cfg.transforms.items()
    train_transform = Compose([instantiate(transform["train_transform"]) for _, transform in transforms])
    inference_transform = Compose([instantiate(transform["inference_transform"]) for _, transform in transforms])
    datamodule = instantiate(cfg.datamodule, train_transform=train_transform, inference_transform=inference_transform)
    log.info(f"{datamodule.__class__.__name__} initialized")
    datamodule.setup()
    log.info(f"Datamodule set up")
    return datamodule


def create_model(cfg: DictConfig, datamodule: LightningDataModule):
    if "Module" in cfg.model._target_:  # DL models
        net_factory = instantiate(cfg.model.net)
        net_params = {"num_classes": datamodule.num_classes} if "Classifier" in cfg.model._target_ else {}
        if "MLP" in cfg.model.net._target_:
            net = net_factory(**net_params, input_size=datamodule.transformed_input_shape[0])
        elif "CNN" in cfg.model.net._target_:
            net = net_factory(**net_params, in_channels=datamodule.transformed_input_shape[0])
        model = instantiate(cfg.model, net=net)
    else:  # ML models
        model = instantiate(cfg.model)
    log.info(f"{model.__class__.__name__} initialized")
    return model


def create_callbacks(cfg: DictConfig):
    callbacks = []
    for name, callback_params in cfg.callbacks.items():
        callbacks.append(instantiate(callback_params))
    return callbacks


def create_trainer(cfg: DictConfig, model, datamodule, logger=None):
    if "Module" in cfg.model._target_:  # DL models
        callbacks = create_callbacks(cfg)
        pl_trainer = instantiate(cfg.trainer.trainer, logger=logger, callbacks=callbacks)
        trainer = instantiate(cfg.trainer, trainer=pl_trainer)(model=model, datamodule=datamodule)
    else:
        trainer = instantiate(cfg.trainer)(model=model, datamodule=datamodule)
    log.info(f"{trainer.__class__.__name__} initialized")
    return trainer


@hydra.main(version_base=None, config_path="../../configs/train_model", config_name="train")
def main(cfg: DictConfig):
    cfg = parse_cfg(cfg)
    print_config_tree(cfg, keys=["datamodule", "model", "callbacks", "logger", "plotter", "transforms"])

    logger = create_logger(cfg)
    datamodule = create_datamodule(cfg)
    model = create_model(cfg, datamodule)

    plotter = instantiate(cfg.plotter)
    log.info(f"{plotter.__class__.__name__} initialized")

    trainer = create_trainer(cfg, model, datamodule, logger)

    log.info(f"Started trainer fit")
    trainer.fit()
    log.info(f"Finished trainer fit")

    dataset, representation_type, model_name = cfg.logger.name.split("__")
    params = dict(dataset=dataset, representation=representation_type, model=model2str[model_name])
    logger.log(params)
    results = trainer.evaluate(plotter=plotter, logger=logger)

    logger.finish()
    log.info("Evaluation finished.")


if __name__ == "__main__":
    main()
