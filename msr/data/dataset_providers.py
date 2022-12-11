import glob
import logging
import math
import os
from abc import abstractmethod
from typing import Dict, List, Literal, Type, Union

import numpy as np
import torch
from joblib import Parallel, delayed
from tqdm.auto import tqdm

from msr.data.download import mimic, ptbxl, sleep_edf
from msr.data.measurements import (
    MimicMeasurement,
    PtbXLMeasurement,
    SleepEDFMeasurement,
)
from msr.data.namespace import get_setters_mask
from msr.signals.representation_extractor import create_representation_extractor

log = logging.getLogger(__name__)


class DatasetProvider:
    def __init__(
        self,
        representation_types: List[str],
        windows_params: Dict[str, Union[str, float, int]],
        fs: float,
        raw_tensors_data_path: str,
        representations_path: str,
        targets_path: str,
    ):
        self.representation_types = representation_types
        self.windows_params = windows_params
        self.fs = fs
        self.raw_tensors_data_path = raw_tensors_data_path
        self.representations_path = representations_path
        self.targets_path = targets_path
        self.representation_extractor_params = dict(windows_params=self.windows_params)

    @property
    @abstractmethod
    def MeasurementFactory(self):
        pass

    def get_representations(self, data, return_feature_names: bool = False, **kwargs):
        measurement = self.MeasurementFactory(*data, fs=self.fs)
        set_beats, set_windows, set_agg_beat = get_setters_mask(self.representation_types)
        rep_extractor = create_representation_extractor(measurement, **self.representation_extractor_params)
        representations = rep_extractor.get_representations(representation_types=self.representation_types, **kwargs)
        if return_feature_names:
            feature_representation_types = [
                rep_type for rep_type in self.representation_types if "features" in rep_type
            ]
            feature_names = rep_extractor.get_feature_names(representation_types=feature_representation_types)
            return representations, feature_names
        return representations

    def create_dataset(
        self, n_jobs=1, batch_size=-1, splits: List[Literal["train", "val", "test"]] = ["train", "val", "test"]
    ):
        def save_feature_names(representations_feature_names: List[str]):
            for rep_type, feature_names in representations_feature_names.items():
                feature_names = np.array(feature_names)
                representation_path = self.representations_path / rep_type
                representation_path.mkdir(parents=True, exist_ok=True)
                np.save(representation_path / f"feature_names.npy", feature_names)

        log.info(f"Creating representation dataset.")
        log.info(f"Data used to create representations is loaded from {self.raw_tensors_data_path}")
        log.info(f"Representations will be stored in {self.representations_path}")
        self.representations_path.mkdir(parents=True, exist_ok=True)
        is_features_in_rep_types = any(["features" in rep_type for rep_type in self.representation_types])

        for i, split in enumerate(tqdm(splits, "Creating representations for all splits")):
            all_data = torch.load(self.raw_tensors_data_path / f"{split}.pt")[:5]
            n_samples = len(all_data)
            if (
                is_features_in_rep_types and i == 0
            ):  # every split has the same features so no need to run it for all splits
                _, representations_feature_names = self.get_representations(all_data[0], return_feature_names=True)
                save_feature_names(representations_feature_names)
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
                    f"Extracting representations for batch {batch_idx + 1} / {n_batches} (samples {start_idx} to"
                    f" {end_idx})"
                )
                if n_jobs == 1:
                    reps = [self.get_representations(data) for data in tqdm(batch_data, desc=desc)]
                else:
                    reps = Parallel(n_jobs=n_jobs)(
                        delayed(self.get_representations)(data) for data in tqdm(batch_data, desc=desc)
                    )
                reps = {
                    rep_type: torch.tensor(np.array([rep[rep_type] for rep in reps]))
                    for rep_type in self.representation_types
                }

                for rep_type, data in reps.items():
                    representation_path = self.representations_path / rep_type
                    representation_path.mkdir(parents=True, exist_ok=True)
                    suffix = f"_{batch_idx}" if n_batches > 1 else ""
                    path = representation_path / f"_{split}{suffix}.pt"
                    torch.save(data, path)
                start_idx = end_idx
            log.info(f"{split} split finished")
        log.info("Representation dataset creation finished.")

    def concat_data_files(self):
        log.info("Concatenating data files.")
        for split in tqdm(["train", "val", "test"], desc="Splits"):
            for rep_type in tqdm(self.representation_types, desc=f"{split} representations"):
                representation_path = self.representations_path / rep_type
                rep_split_files = sorted(glob.glob(f"{str(representation_path)}/{split}_*.pt"))
                if len(rep_split_files) > 0:
                    torch.save(
                        torch.concatenate([torch.load(file) for file in rep_split_files]),
                        representation_path / f"{split}.pt",
                    )
                    for filepath in rep_split_files:
                        os.remove(filepath)

    def load_split(self, split, representation_type):
        representation_path = self.representations_path / representation_type
        data = torch.load(representation_path / f"{split}.pt")
        targets = torch.load(self.targets_path / f"{split}.pt")
        try:
            info = np.load(self.targets_path / "info.npy", allow_pickle=True)
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


class PeriodicDatasetProvider(DatasetProvider):
    def __init__(
        self,
        representation_types: List[str],
        windows_params: Dict[str, Union[str, float, int]],
        beats_params: Dict[str, Union[str, float, int]],
        agg_beat_params: Dict[str, Union[str, float, int]],
        n_beats: int,
        fs: float,
        raw_tensors_data_path: str,
        representations_path: str,
        targets_path: str,
    ):
        super().__init__(
            representation_types=representation_types,
            windows_params=windows_params,
            fs=fs,
            raw_tensors_data_path=raw_tensors_data_path,
            representations_path=representations_path,
            targets_path=targets_path,
        )
        self.beats_params = beats_params
        self.agg_beat_params = agg_beat_params
        self.representation_extractor_params.update(
            {
                "beats_params": self.beats_params,
                "agg_beat_params": self.agg_beat_params,
            }
        )
        self.n_beats = n_beats

    def get_representations(self, data, return_feature_names: bool = False):
        return super(PeriodicDatasetProvider, self).get_representations(
            data, return_feature_names, n_beats=self.n_beats
        )


class MimicDatasetProvider(PeriodicDatasetProvider):
    def __init__(
        self,
        representation_types: List[str],
        windows_params: Dict[str, Union[str, float, int]],
        beats_params: Dict[str, Union[str, float, int]],
        agg_beat_params: Dict[str, Union[str, float, int]],
        n_beats: int,
        target: str,
    ):
        super().__init__(
            representation_types=representation_types,
            windows_params=windows_params,
            beats_params=beats_params,
            agg_beat_params=agg_beat_params,
            n_beats=n_beats,
            fs=mimic.FS,
            raw_tensors_data_path=mimic.RAW_TENSORS_DATA_PATH,
            representations_path=mimic.DATASET_PATH / "representations",
            targets_path=mimic.RAW_TENSORS_TARGETS_PATH / target,
        )

    @property
    def MeasurementFactory(self):
        return MimicMeasurement

    def get_representations(self, data, return_feature_names: bool = False):
        return super(MimicDatasetProvider, self).get_representations(data.T.numpy(), return_feature_names)


class PtbXLDatasetProvider(PeriodicDatasetProvider):
    def __init__(
        self,
        representation_types: List[str],
        windows_params: Dict[str, Union[str, float, int]],
        beats_params: Dict[str, Union[str, float, int]],
        agg_beat_params: Dict[str, Union[str, float, int]],
        n_beats: int,
        target: str,
    ):
        super().__init__(
            representation_types=representation_types,
            windows_params=windows_params,
            beats_params=beats_params,
            agg_beat_params=agg_beat_params,
            n_beats=n_beats,
            fs=ptbxl.FS,
            raw_tensors_data_path=ptbxl.RAW_TENSORS_DATA_PATH,
            representations_path=ptbxl.DATASET_PATH / f"representations_{ptbxl.FS}",
            targets_path=ptbxl.RAW_TENSORS_TARGETS_PATH / target,
        )

    @property
    def MeasurementFactory(self):
        return PtbXLMeasurement

    def get_representations(self, data, return_feature_names: bool = False):
        return super(PtbXLDatasetProvider, self).get_representations(data.T.numpy(), return_feature_names)


class SleepEDFDatasetProvider(DatasetProvider):
    def __init__(
        self,
        representation_types: List[str],
        windows_params: Dict[str, Union[str, float, int]],
    ):
        super().__init__(
            representation_types=representation_types,
            windows_params=windows_params,
            fs=sleep_edf.FS,
            raw_tensors_data_path=sleep_edf.RAW_TENSORS_DATA_PATH,
            representations_path=sleep_edf.DATASET_PATH / f"representations",
            targets_path=sleep_edf.RAW_TENSORS_TARGETS_PATH,
        )

    @property
    def MeasurementFactory(self):
        return SleepEDFMeasurement

    def get_representations(self, data, return_feature_names: bool = False):
        return super(SleepEDFDatasetProvider, self).get_representations(data.numpy(), return_feature_names)
