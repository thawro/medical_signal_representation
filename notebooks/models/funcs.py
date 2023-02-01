import pytorch_lightning as pl
from tqdm.auto import tqdm

from msr.training.data.datamodules import PtbXLDataModule
from msr.training.data.transforms import Flatten
from msr.training.trainers import (
    DLClassifierTrainer,
    DLRegressorTrainer,
    MLClassifierTrainer,
    MLRegressorTrainer,
)

REP_TYPES = ["whole_signal_waveforms", "whole_signal_features", "agg_beat_waveforms", "agg_beat_features"]


def run_ml_experiment(rep_type, model_provider, task="clf"):
    ml_transform = Flatten(start_dim=0, end_dim=-1)
    dm = PtbXLDataModule(
        representation_type=rep_type,
        target="diagnostic_class",
        train_transform=ml_transform,
        inference_transform=ml_transform,
    )
    dm.setup()
    model = model_provider()
    # model = DecisionTreeClassifier()
    trainer = MLClassifierTrainer(model, dm) if task == "clf" else MLRegressorTrainer(model, dm)
    trainer.fit()
    results = trainer.evaluate()
    return results


def run_dl_experiment(rep_type, model_provider, train_transform, inference_transform, task="clf"):
    ml_transform = Flatten(start_dim=0, end_dim=-1)

    dm = PtbXLDataModule(
        representation_type=rep_type,
        target="diagnostic_class",
        train_transform=train_transform,
        inference_transform=inference_transform,
    )
    dm.setup()
    model = model_provider(num_classes=dm.num_classes, input_shape=dm.input_shape)
    trainer = pl.Trainer(
        # logger=WandbLogger(project=project_name, name=run_name, id=logger.id),
        accelerator="auto",
        max_epochs=10,
    )

    TrainerClass = DLClassifierTrainer if task == "clf" else DLRegressorTrainer(model, dm)
    dl_trainer = TrainerClass(trainer, model, dm)
    dl_trainer.fit()
    results = dl_trainer.evaluate()
    return results


def run_all_ml_experiments(model_provider, rep_types=REP_TYPES, task="clf"):
    results = {}
    for rep_type in tqdm(rep_types):
        rep_results = run_ml_experiment(rep_type, model_provider, task)
        results[rep_type] = rep_results["metrics"]
        print(rep_type, round(rep_results["metrics"]["val/auroc"], 2))
    return results
