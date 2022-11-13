import omegaconf

import wandb

PROJECT = "medical-signals-representation"


class MLWandbLogger:
    def __init__(self, project: str, run_name: str = None, entity: str = None):
        self.project = project
        self.run_name = run_name
        self.entity = entity

    def init(self, config: omegaconf.DictConfig = None):
        wandb.init(
            name=self.run_name,
            entity=self.entity,
            project=self.project,
            config=omegaconf.OmegaConf.to_container(config, resolve=True, throw_on_missing=True),
        )

    def log(self, data, commit=False):
        wandb.log(data, commit=commit)

    def finish(self, quiet: bool = True):
        wandb.finish(quiet=quiet)
