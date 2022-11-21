import omegaconf

import wandb

PROJECT = "medical-signals-representation"


class MLWandbLogger:
    def __init__(self, project: str, run_name: str = None):
        self.project = project
        self.run_name = run_name

    @property
    def id(self):
        return wandb.run.id

    def init(self, config: omegaconf.DictConfig = None):
        cfg = (
            omegaconf.OmegaConf.to_container(config, resolve=True, throw_on_missing=True)
            if config is not None
            else config
        )

        wandb.init(
            name=self.run_name,
            project=self.project,
            config=cfg,
        )

    def log(self, data, commit=False):
        wandb.log(data, commit=commit)

    def finish(self, quiet: bool = True):
        wandb.finish(quiet=quiet)
