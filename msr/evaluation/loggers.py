import wandb

PROJECT = "medical-signals-representation"


class BaseWandbLogger:
    def __init__(self, project: str, run_name: str = None):
        self.project = project
        self.run_name = run_name
        wandb.init(project=project, name=run_name)

    def log(self, data, commit=False):
        wandb.log(data, commit=commit)

    def finish(self, quiet: bool = True):
        wandb.finish(quiet=quiet)
