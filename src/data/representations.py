from typing import Dict, List, Literal, Type

import numpy as np
import torch
from joblib import Parallel, delayed
from tqdm.auto import tqdm

from data.utils import PeriodicRepresentation, Representation, get_setters_mask
from signals.base import MultiChannelPeriodicSignal, MultiChannelSignal
from signals.representation_extractor import (
    PeriodicRepresentationExtractor,
    RepresentationExtractor,
)


def get_periodic_representations(
    multichannel_signal: Type[MultiChannelPeriodicSignal],
    beats_params: dict,
    agg_beat_params: dict,
    windows_params: dict,
    representation_types: List[str],
) -> Dict[str, torch.Tensor]:
    rep_extractor = PeriodicRepresentationExtractor(multichannel_signal)
    set_beats, set_windows, set_agg_beat = get_setters_mask(representation_types)
    if set_beats:
        rep_extractor.set_beats(**beats_params)
        if set_agg_beat:
            rep_extractor.set_agg_beat(**agg_beat_params)
    if set_windows:
        rep_extractor.set_windows(**windows_params)
    return rep_extractor.get_representations(representation_types=representation_types)


def get_representations(
    multichannel_signal: Type[MultiChannelSignal], windows_params: dict, representation_types: List[str]
) -> Dict[str, torch.Tensor]:
    _, set_windows, _ = get_setters_mask(representation_types)
    rep_extractor = RepresentationExtractor(multichannel_signal)
    if set_windows:
        rep_extractor.set_windows(**windows_params)
    return rep_extractor.get_representations(representation_types=representation_types)


def create_dataset(raw_tensors_path, representations_path, get_repr_func, use_multiprocessing=False, **kwargs):
    representations_path.mkdir(parents=True, exist_ok=True)
    splits = ["train", "val", "test"]
    for split in tqdm(splits, "Creating representations for all splits"):
        all_data = torch.load(raw_tensors_path / f"{split}_data.pt")
        if use_multiprocessing:
            reps = Parallel(n_jobs=-1)(
                delayed(get_repr_func)(data, **kwargs) for data in tqdm(all_data, desc="Extracting representations")
            )
        else:
            reps = [get_repr_func(data, **kwargs) for data in tqdm(all_data, desc="Extracting representations")]
        rep_names = reps[0].keys()
        reps = {name: torch.tensor(np.array([rep[name] for rep in reps])) for name in rep_names}
        for name, data in reps.items():
            representation_path = representations_path / name
            representation_path.mkdir(parents=True, exist_ok=True)
            path = representation_path / f"{split}_data.pt"
            torch.save(data, path)


def load_split(split, representations_path, targets_path, representation_type):
    data = torch.load(representations_path / f"{representation_type}/{split}_data.pt")
    targets = np.load(targets_path / f"{split}_targets.npy")
    info = np.load(targets_path / "info.npy", allow_pickle=True)
    info = {i: info[i] for i in range(len(info))}
    return {"data": data, "targets": targets, "info": info}
