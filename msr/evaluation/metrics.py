import os
import pickle
import time
from functools import partial
from typing import Callable, Dict, List

import numpy as np
import torch
from torchmetrics.functional import (
    accuracy,
    auroc,
    f1_score,
    mean_absolute_error,
    mean_absolute_percentage_error,
    mean_squared_error,
    pearson_corrcoef,
    r2_score,
    roc,
)
from tqdm.auto import tqdm


def get_metrics(metrics_funcs: Dict[str, Callable], preds: torch.Tensor, target: torch.Tensor, metrics: List[str]):
    preds = torch.from_numpy(preds) if isinstance(preds, np.ndarray) else preds
    target = torch.from_numpy(target) if isinstance(target, np.ndarray) else target
    calculated_metrics = {}
    for name, metric in metrics_funcs.items():
        if name not in metrics:
            continue
        if name in ["roc"]:
            fpr, tpr, threshold = metric(preds, target)
            calculated_metrics[name] = fpr, tpr, threshold
        else:
            calculated_metrics[name] = metric(preds, target).item()
    return calculated_metrics


def get_classification_metrics(num_classes, preds, target, metrics):
    classification_metrics = dict(
        accuracy=partial(accuracy, num_classes=num_classes),
        fscore=partial(f1_score, num_classes=num_classes, average="weighted"),
        auroc=partial(auroc, num_classes=num_classes, average="weighted"),
        roc=partial(roc, num_classes=num_classes),
    )
    return get_metrics(metrics_funcs=classification_metrics, preds=preds, target=target, metrics=metrics)


def get_regression_metrics(preds, target, metrics):
    regression_metrics = dict(
        mae=mean_absolute_error,
        mape=mean_absolute_percentage_error,
        corr=pearson_corrcoef,
        r2=r2_score,
        mse=mean_squared_error,
    )
    return get_metrics(metrics_funcs=regression_metrics, preds=preds, target=target, metrics=metrics)


def get_dl_computational_complexity(model, dummy_input, n_iter=300):
    starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    timings = np.zeros((n_iter, 1))
    # GPU-WARM-UP
    for _ in range(10):
        _ = model(dummy_input)
    # MEASURE PERFORMANCE
    with torch.no_grad():
        for rep in tqdm(range(n_iter), desc="Measuring computational complexity"):
            starter.record()
            _ = model(dummy_input)
            ender.record()
            # WAIT FOR GPU SYNC
            torch.cuda.synchronize()
            curr_time = starter.elapsed_time(ender)
            timings[rep] = curr_time
    mean = np.sum(timings) / n_iter
    std = np.std(timings)
    return mean, std


def get_ml_computational_complexity(model, dummy_input, n_iter=300):
    timings = np.zeros((n_iter, 1))
    for _ in range(10):
        _ = model.predict(dummy_input)
    for rep in tqdm(range(n_iter), desc="Measuring computational complexity"):
        start = time.time()
        _ = model.predict(dummy_input)
        end = time.time()
        duration = end - start  # in seconds
        duration = duration * 1000  # in miliseconds
        timings[rep] = duration
    mean = np.sum(timings) / n_iter
    std = np.std(timings)
    return mean, std


def get_memory_complexity(model, is_dl):
    if is_dl:
        path = "model.pt"
        torch.save(model, path)
    else:
        path = "model.pickle"
        pickle.dump(model, open(path, "wb"))
    size = os.path.getsize(path)
    os.remove(path)
    return size
