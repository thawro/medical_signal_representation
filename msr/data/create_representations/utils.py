import glob
import inspect
import logging
import math
import os
from typing import Dict, List, Type

import numpy as np
import torch
from joblib import Parallel, delayed
from tqdm.auto import tqdm

from msr.data.namespace import get_setters_mask
from msr.signals.base import MultiChannelPeriodicSignal, MultiChannelSignal
from msr.signals.representation_extractor import (
    PeriodicRepresentationExtractor,
    RepresentationExtractor,
)
from msr.utils import run_in_parallel_with_joblib

log = logging.getLogger(__name__)


def create_periodic_representation_extractor(
    multichannel_signal: Type[MultiChannelPeriodicSignal],
    beats_params: dict,
    agg_beat_params: dict,
    windows_params: dict,
):
    rep_extractor = PeriodicRepresentationExtractor(multichannel_signal)
    if windows_params:
        rep_extractor.set_windows(**windows_params)
    if beats_params:
        rep_extractor.set_beats(**beats_params)
        if agg_beat_params:
            rep_extractor.set_agg_beat(**agg_beat_params)
    return rep_extractor


def create_representation_extractor(
    multichannel_signal: Type[MultiChannelPeriodicSignal],
    windows_params: dict,
):
    rep_extractor = RepresentationExtractor(multichannel_signal)
    if windows_params:
        rep_extractor.set_windows(**windows_params)
    return rep_extractor


def get_periodic_representations(
    multichannel_signal: Type[MultiChannelPeriodicSignal],
    representation_types: List[str],
    beats_params: dict,
    n_beats: int,
    agg_beat_params: dict,
    windows_params: dict,
    return_feature_names: bool = False,
) -> Dict[str, torch.Tensor]:
    set_beats, set_windows, set_agg_beat = get_setters_mask(representation_types)
    rep_extractor = create_periodic_representation_extractor(
        multichannel_signal,
        beats_params if set_beats else {},
        agg_beat_params if set_agg_beat else {},
        windows_params if set_windows else {},
    )
    representations = rep_extractor.get_representations(representation_types=representation_types, n_beats=n_beats)
    if return_feature_names:
        feature_names = rep_extractor.get_feature_names(representation_types=representation_types)
        return representations, feature_names
    else:
        return representations


def get_representations(
    multichannel_signal: Type[MultiChannelSignal],
    windows_params: dict,
    representation_types: List[str],
    return_feature_names: bool = False,
) -> Dict[str, torch.Tensor]:
    _, set_windows, _ = get_setters_mask(representation_types)
    rep_extractor = create_representation_extractor(multichannel_signal, windows_params if set_windows else {})
    representations = rep_extractor.get_representations(representation_types=representation_types)
    if return_feature_names:
        feature_names = rep_extractor.get_feature_names(representation_types=representation_types)
        return representations, feature_names
    else:
        return representations


def save_feature_names(path, representations_feature_names):
    for rep_type, feature_names in representations_feature_names.items():
        feature_names = np.array(feature_names)
        representation_path = path / rep_type
        representation_path.mkdir(parents=True, exist_ok=True)
        np.save(representation_path / f"feature_names.npy", feature_names)


def create_representations_dataset(raw_tensors_path, representations_path, get_repr_func, n_jobs=1, batch_size=-1):
    log.info(f"Creating representation dataset.")
    log.info(f"Data used to create representations is loaded from {raw_tensors_path}")
    log.info(f"Representations will be stored in {representations_path}")
    representations_path.mkdir(parents=True, exist_ok=True)
    splits = ["train", "val", "test"]
    # splits = ["test"]
    rep_types = inspect.signature(get_repr_func).parameters["representation_types"].default
    is_features_in_rep_types = any(["features" in rep_type for rep_type in rep_types])

    for i, split in enumerate(tqdm(splits, "Creating representations for all splits")):
        all_data = torch.load(raw_tensors_path / f"{split}.pt")
        n_samples = len(all_data)
        if is_features_in_rep_types and i == 0:  # every split has the same features so no need to run it for all splits
            _, representations_feature_names = get_repr_func(all_data[0], return_feature_names=True)
            save_feature_names(representations_path, representations_feature_names)
            log.info("Feature names saved")

        _batch_size = n_samples if batch_size == -1 else batch_size
        start_idx = 0
        n_batches = math.ceil(n_samples / _batch_size)
        log.info(f"Exctracting representations using batch_size={_batch_size} ({n_batches} batches)")
        while start_idx < n_samples:
            end_idx = start_idx + _batch_size
            batch_idx = math.ceil((start_idx + 1) / _batch_size) - 1
            batch_data = all_data[start_idx:end_idx]
            desc = (
                f"Extracting representations for batch {batch_idx + 1} / {n_batches} (samples {start_idx} to {end_idx})"
            )
            if n_jobs == 1:
                reps = [get_repr_func(data) for data in tqdm(batch_data, desc=desc)]
            else:
                reps = Parallel(n_jobs=n_jobs)(delayed(get_repr_func)(data) for data in tqdm(batch_data, desc=desc))
            reps = {rep_type: torch.tensor(np.array([rep[rep_type] for rep in reps])) for rep_type in rep_types}

            for rep_type, data in reps.items():
                representation_path = representations_path / rep_type
                representation_path.mkdir(parents=True, exist_ok=True)
                suffix = f"_{batch_idx}" if n_batches > 1 else ""
                path = representation_path / f"{split}{suffix}.pt"
                torch.save(data, path)
            start_idx = end_idx
        log.info(f"{split} split finished")
    log.info("Representation dataset creation finished.")


def concat_data_files(representations_path, representation_types):
    log.info("Concatenating data files.")
    for split in tqdm(["train", "val", "test"], desc="Splits"):
        for rep_type in tqdm(representation_types, desc=f"{split} representations"):
            representation_path = representations_path / rep_type
            rep_split_files = sorted(glob.glob(f"{str(representation_path)}/{split}_*.pt"))
            if len(rep_split_files) > 0:
                torch.save(
                    torch.concatenate([torch.load(file) for file in rep_split_files]),
                    representation_path / f"{split}.pt",
                )
                for filepath in rep_split_files:
                    os.remove(filepath)


def load_split(split, representations_path, targets_path, representation_type):
    representation_path = representations_path / representation_type
    data = torch.load(representation_path / f"{split}.pt")
    targets = torch.load(targets_path / f"{split}.pt")
    try:
        info = np.load(targets_path / "info.npy", allow_pickle=True)
        info = {i: info[i] for i in range(len(info))}
    except FileNotFoundError:
        log.warning("No info file for that split")
        info = {}
    split_data = {"data": data, "targets": targets, "info": info}
    try:
        feature_names = np.load(representation_path / "feature_names.npy")
        split_data["feature_names"] = feature_names
    except Exception:  # TODO: Change to sth like Not file found
        pass
    return split_data
