import omegaconf

import wandb

PROJECT = "medical-signals-representation"


class MLWandbLogger:
    def __init__(self, project: str, name: str = None, **kwargs):
        self.project = project
        self.name = name
        self.kwargs = kwargs

    @property
    def id(self):
        return wandb.run.id

    def init(self, config: omegaconf.DictConfig = None):
        cfg = (
            omegaconf.OmegaConf.to_container(config, resolve=True, throw_on_missing=True)
            if config is not None
            else config
        )

        wandb.init(name=self.name, project=self.project, config=cfg, **self.kwargs)

    def log(self, data, commit=False):
        wandb.log(data, commit=commit)

    def finish(self, quiet: bool = True):
        wandb.finish(quiet=quiet)
