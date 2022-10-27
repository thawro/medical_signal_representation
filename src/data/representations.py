from typing import Dict, Type

import numpy as np
import torch

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
) -> Dict[str, torch.Tensor]:
    rep_extractor = PeriodicRepresentationExtractor(multichannel_signal)
    rep_extractor.set_beats(**beats_params)
    rep_extractor.set_agg_beat(**agg_beat_params)
    rep_extractor.set_windows(**windows_params)
    return rep_extractor.get_representations()


def get_representations(multichannel_signal: Type[MultiChannelSignal], windows_params: dict) -> Dict[str, torch.Tensor]:
    rep_extractor = RepresentationExtractor(multichannel_signal)
    rep_extractor.set_windows(**windows_params)
    return rep_extractor.get_representations()


def create_dataset(raw_tensors_path, representations_path, get_repr_func, **kwargs):
    representations_path.mkdir(parents=True, exist_ok=True)
    splits = ["train", "val", "test"]
    for split in tqdm(splits, "Creating representations for all splits"):
        all_data = torch.load(raw_tensors_path / f"{split}_data.pt")
        reps = Parallel(n_jobs=-1)(delayed(get_repr_func)(data, **kwargs) for data in tqdm(all_data))
        rep_names = reps[0].keys()
        reps = {name: torch.tensor(np.array([rep[name] for rep in reps])) for name in rep_names}
        for name, data in reps.items():
            path = representations_path / name / f"{split}_data.pt"
            torch.save(data, path)


def load_split(split, representations_path, targets_path, representation_type):
    data = torch.load(representations_path / f"{representation_type}/{split}_data.pt")
    targets = np.load(targets_path / f"{split}_targets.npy")
    info = np.load(targets_path / "info.npy", allow_pickle=True)
    info = {i: info[i] for i in range(len(info))}
    return {"data": data, "targets": targets, "info": info}
