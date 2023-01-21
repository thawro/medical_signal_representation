import logging
import os
import time
from pathlib import Path

import hydra
import wandb
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning import LightningDataModule
from pytorch_lightning.utilities.seed import seed_everything
from thop import profile
from torchinfo import summary
from torchvision.transforms import Compose

from msr.evaluation.metrics import (
    get_dl_computational_complexity,
    get_memory_complexity,
    get_ml_computational_complexity,
)
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
    if "transforms" in cfg:
        transforms = cfg.transforms.items()
        train_transform = Compose([instantiate(transform["train_transform"]) for _, transform in transforms])
        inference_transform = Compose([instantiate(transform["inference_transform"]) for _, transform in transforms])
    else:
        train_transform = None
        inference_transform = None
    datamodule = instantiate(cfg.datamodule, train_transform=train_transform, inference_transform=inference_transform)
    log.info(f"{datamodule.__class__.__name__} initialized")
    datamodule.setup(stage="fit")
    log.info(f"Datamodule set up")
    return datamodule


def create_model(cfg: DictConfig, datamodule: LightningDataModule, ckpt_path=None):
    if "Module" in cfg.model._target_:  # DL models
        net_factory = instantiate(cfg.model.net)
        net_params = {"num_classes": datamodule.num_classes} if "Classifier" in cfg.model._target_ else {}
        if "MLP" in cfg.model.net._target_:
            net = net_factory(**net_params, input_size=datamodule.transformed_input_shape[0])
        elif "CNN" in cfg.model.net._target_:
            net = net_factory(**net_params, in_channels=datamodule.transformed_input_shape[0])
        elif "LSTM" in cfg.model.net._target_:
            net = net_factory(**net_params, in_dim=datamodule.transformed_input_shape[1])

        model = instantiate(cfg.model, net=net)
        if ckpt_path is not None:
            log.info("Loading model from checkpoint path")
            model = model.__class__.load_from_checkpoint(ckpt_path)
    else:  # ML models
        model = instantiate(cfg.model)
    log.info(model)
    log.info(f"{model.__class__.__name__} initialized")
    return model


def create_callbacks(cfg: DictConfig):
    callbacks = {}
    for name, callback_params in cfg.callbacks.items():
        callbacks[name] = instantiate(callback_params)
    return callbacks


def create_trainer(cfg: DictConfig, model, datamodule, logger=None, callbacks=None):
    if "Module" in cfg.model._target_:  # DL models
        pl_trainer = instantiate(cfg.trainer.trainer, logger=logger, callbacks=callbacks)
        trainer = instantiate(cfg.trainer, trainer=pl_trainer)(model=model, datamodule=datamodule)
    else:
        trainer = instantiate(cfg.trainer)(model=model, datamodule=datamodule)
    log.info(f"{trainer.__class__.__name__} initialized")
    return trainer


def log_model(model, dummy_input):
    model_representations = [(f"model_modules.txt", str(model))]
    col_names = ["input_size", "output_size", "kernel_size", "num_params"]
    model_summary = summary(
        model, input_data=dummy_input, depth=10, verbose=0, device=model.device, col_names=col_names
    )
    model_representations.append((f"model_summary.txt", str(model_summary)))
    model_dir_path = Path(os.path.join(wandb.run.dir, "model"))
    model_dir_path.mkdir(parents=True, exist_ok=True)
    for path, model_str in model_representations:
        text_file = open(str(model_dir_path / path), "w")
        text_file.write(model_str)
        text_file.close()
        wandb.save(f"model/{path}")


@hydra.main(version_base=None, config_path="../../configs/train_model", config_name="train")
def main(cfg: DictConfig):
    cfg = parse_cfg(cfg)
    is_dl = "Module" in cfg.model._target_  # DL models
    print_config_tree(cfg, keys=["datamodule", "model", "callbacks", "logger", "plotter", "transforms"])
    seed_everything(cfg.seed, workers=True)
    logger = create_logger(cfg)
    datamodule = create_datamodule(cfg)
    model = create_model(cfg, datamodule)

    callbacks_lst = None
    dummy_input = datamodule.train[0][0].unsqueeze(0)

    if is_dl:
        log_model(model, dummy_input=dummy_input)
        callbacks = create_callbacks(cfg)
        callbacks_lst = list(callbacks.values())

    dataset, representation_type, model_name = cfg.logger.name.split("__")
    logged_params = dict(
        dataset=dataset,
        representation=representation_type,
        model=model2str[model_name],
    )
    wandb.log(logged_params, commit=True)

    plotter = instantiate(cfg.plotter)
    log.info(f"{plotter.__class__.__name__} initialized")

    trainer = create_trainer(cfg, model, datamodule, logger, callbacks_lst)

    log.info(f"Started trainer fit")
    start = time.time()
    trainer.fit()
    fit_time = time.time() - start
    log.info(f"Finished trainer fit")

    datamodule.train = None
    datamodule.setup(stage="test")
    # Loading best model
    if is_dl:
        best_ckpt_path = callbacks["model_checkpoint"].best_model_path
        model = create_model(cfg, datamodule, best_ckpt_path)
        model.eval()
        trainer = create_trainer(cfg, model, datamodule, logger, callbacks=None)

    results = trainer.evaluate(plotter=plotter)

    log.info("Started measuring memory and computational complexity")
    memory_complexity = get_memory_complexity(model, is_dl)
    if is_dl:
        inference_mean_time, inference_std_time = get_dl_computational_complexity(model, dummy_input, n_iter=300)
        model_macs, model_params = profile(model, inputs=(dummy_input,))
    else:
        inference_mean_time, inference_std_time = get_ml_computational_complexity(model, dummy_input, n_iter=300)
        model_macs, model_params = -1, -1
    log.info("Finished measuring memory and computational complexity")

    complexity_params = dict(
        fit_time=fit_time,
        memory_complexity=memory_complexity,
        inference_mean_time=inference_mean_time,
        inference_std_time=inference_std_time,
        model_macs=model_macs,
        model_params=model_params,
    )
    print(complexity_params)
    wandb.log(complexity_params)
    wandb.log({"fit_time": fit_time})

    wandb.finish()
    log.info("Evaluation finished.")


if __name__ == "__main__":
    main()
