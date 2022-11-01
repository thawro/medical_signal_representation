import glob
import logging
import multiprocessing
import os
import random
from functools import partial
from itertools import groupby

import hydra
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests
import torch
import wfdb
from joblib import Parallel, delayed
from omegaconf import DictConfig, OmegaConf
from requests.adapters import HTTPAdapter
from requests.packages.urllib3.util.retry import Retry
from tqdm.auto import tqdm

from msr.data.raw.utils import validate_signal
from msr.data.utils import create_train_val_test_split_info
from msr.utils import DATA_PATH, append_txt_to_file, print_info

log = logging.getLogger(__name__)
log.setLevel(logging.INFO)

MIMIC_DB_NAME = "mimic3wdb-matched"  # specify type of mimic databes (mimic3wdb or mimic3wdb-matched)
MIMIC_URL = f"https://physionet.org/files/{MIMIC_DB_NAME}/1.0"
MIMIC_PATH = DATA_PATH / "mimic"
MIMIC_DB_PATH = MIMIC_PATH / "db"  # raw data in .dat and .hea formats
LOGS_PATH = MIMIC_PATH / "logs"

SEGMENTS_FILE_PATH = LOGS_PATH / "segments_info.txt"  # file for segments info
SAMPLE_SEGMENTS_FILE_PATH = LOGS_PATH / "sample_segments_info.txt"  # filename for samples segments info
SPLIT_INFO_PATH = LOGS_PATH / "split_info.csv"

RAW_DATASET_PATH = MIMIC_PATH / "raw"  # path to directory with csv files
RAW_TENSORS_PATH = MIMIC_PATH / "raw_tensors"
RAW_TENSORS_DATA_PATH = RAW_TENSORS_PATH / "data"
TARGETS_PATH = RAW_TENSORS_PATH / "targets"
SESSION = None


def set_global_session():
    global SESSION
    if not SESSION:
        SESSION = requests.Session()
        retry = Retry(connect=100, total=100, backoff_factor=0.1)
        adapter = HTTPAdapter(max_retries=retry)
        SESSION.mount("http://", adapter)
        SESSION.mount("https://", adapter)


def get_measurements_paths_for_single_subject(subject):
    """Return measurements paths for single subject using RECORDS file in physionet mimic3db database."""
    url = f"{MIMIC_URL}/{subject}/RECORDS"
    with SESSION.get(url) as response:
        subject_measurements = response.text.split("\n")[:-1]
        subject_measurements = [
            f"{subject}/{measurement}" for measurement in subject_measurements if "-" not in measurement
        ]
        append_txt_to_file(LOGS_PATH / "measurements_paths.txt", subject_measurements)
        return subject_measurements


def check_signals_availability_for_single_measurement(measurement_path, sig_names):
    """Check if all desired signals are available (in the header file)."""
    chunksize = 1024 * 1024
    directory, subject, measurement = measurement_path.split("/")
    url_hea = f"{MIMIC_URL}/{directory}/{subject}/{measurement}.hea"
    tmp_path = MIMIC_PATH / "tmp"
    tmp_path.mkdir(parents=True, exist_ok=True)
    hea_filename = tmp_path / f"{measurement}.hea"
    try:
        with SESSION.get(url_hea, stream=True) as response:
            response.raise_for_status()
            with open(hea_filename, "wb") as f:
                for chunk in response.iter_content(chunk_size=chunksize):
                    f.write(chunk)
    except requests.exceptions.HTTPError as e:  # No URL found
        log.error(f"{measurement_path}: {e}")
        return {
            "path": measurement_path,
            **{sig_name: False for sig_name in sig_names},
            "n_samples": 0,
        }
    header = wfdb.rdheader(str(tmp_path / measurement))
    os.remove(hea_filename)

    signals_availability_txt = ",".join([str(sig_name in header.sig_name) for sig_name in sig_names])
    row_txt = f"{measurement_path},{signals_availability_txt},{header.sig_len}"
    append_txt_to_file(LOGS_PATH / "measurements_signals_availability.txt", row_txt)

    return {
        "path": measurement_path,
        **{sig_name: sig_name in header.sig_name for sig_name in sig_names},
        "n_samples": header.sig_len,
    }


def download_file(url, filename, chunksize=1024 * 1024):
    with SESSION.get(url, stream=True) as response:
        response.raise_for_status()
        with open(filename, "wb") as f:
            for chunk in response.iter_content(chunk_size=chunksize):
                f.write(chunk)


def download_single_measurement(measurement_path):
    """Download measurement from physionet database."""
    chunksize = 1024 * 1024
    directory, subject, measurement = measurement_path.split("/")
    path = MIMIC_DB_PATH / directory / subject
    path.mkdir(parents=True, exist_ok=True)
    dat_url = f"{MIMIC_URL}/{directory}/{subject}/{measurement}.dat"
    hea_url = f"{MIMIC_URL}/{directory}/{subject}/{measurement}.hea"
    dat_filename = f"{path}/{measurement}.dat"
    hea_filename = f"{path}/{measurement}.hea"
    download_file(dat_url, dat_filename, chunksize)
    download_file(hea_url, hea_filename, chunksize)
    append_txt_to_file(LOGS_PATH / "downloaded_measurements.txt", measurement_path)
    log.info(f"{measurement_path}: downloaded")


def load_data_to_array(measurement_path, sig_names):
    """Load data to DataFrame using wfdb library"""
    try:
        data, info = wfdb.rdsamp(str(MIMIC_DB_PATH / measurement_path))
    except ValueError as e:
        log.error(f"{measurement_path}: {e}. mimic db file was corrupted. Downloading it again")
        with multiprocessing.Pool(initializer=set_global_session, processes=1) as pool:
            for _ in pool.imap_unordered(download_single_measurement, [measurement_path]):
                pass
        data, info = wfdb.rdsamp(str(MIMIC_DB_PATH / measurement_path))
    sample_sig_names = info["sig_name"]
    sig_idxs = [sample_sig_names.index(sig_name) for sig_name in sig_names]
    return data[:, sig_idxs]  # data of shape [N, C], where C are signals from sig_names


def validate_recording(abp, ppg, ecg, fs, sample_len_sec, max_n_same, plot=False):
    """Validate recordings all signals (ABP, PPG and ECG).
    Returns list[bool] where True indicates samples which are good for all validation steps for all signals.
    """
    valid_sig_len = len(abp) >= sample_len_sec * fs

    if valid_sig_len or plot:
        abp_is_good = validate_signal(abp, low=0, high=300, max_n_same=max_n_same)
        ppg_is_good = validate_signal(ppg, low=None, high=None, max_n_same=max_n_same)
        ecg_is_good = validate_signal(ecg, low=None, high=None, max_n_same=max_n_same)
        is_good = abp_is_good & ppg_is_good & ecg_is_good
    else:
        is_good = np.zeros_like(abp) == 1  # not enough samples, all False

    if plot:
        fig, axes = plt.subplots(3, 1, figsize=(24, 18))
        signals = [("ABP", abp, abp_is_good), ("PPG", ppg, ppg_is_good), ("ECG", ecg, ecg_is_good)]
        t = np.arange(len(abp)) / fs
        for (name, sig, sig_is_good), ax in zip(signals, axes):
            ax.plot(t, sig, lw=2)
            ax.scatter(t[sig_is_good], sig[sig_is_good], color="green", s=10)
            ax.scatter(t[~sig_is_good], sig[~sig_is_good], color="red", s=10)
            percent_good = sig_is_good.sum() / len(sig_is_good) * 100
            min_v, max_v = np.nanmin(sig), np.nanmax(sig)
            ax.fill_between(t, min_v, max_v, where=~is_good, color="red", alpha=0.07)
            ax.fill_between(t, min_v, max_v, where=~sig_is_good, color="red", alpha=0.1)
            ax.set_title(f"{name}: Percent of good signal: {round(percent_good, 2)}%", fontsize=22)
    return is_good


def find_valid_segments(abp, ppg, ecg, fs, sample_len_sec, max_n_same, plot=False):
    """Find valid segments for given ABP, PPG and ECG signals using validate_recording func."""
    is_good = validate_recording(abp, ppg, ecg, fs, max_n_same, sample_len_sec, plot=plot)
    segments = [
        (group[0], group[-1])
        for group in (list(group) for key, group in groupby(range(len(is_good)), key=is_good.__getitem__) if key)
    ]
    good_segments = [segment for segment in segments if (segment[1] - segment[0]) / fs > sample_len_sec]
    if plot:
        is_good = np.zeros_like(abp) == 1
        for start, end in good_segments:
            is_good[start:end] = True
        fig, axes = plt.subplots(3, 1, figsize=(24, 18))
        signals = [("ABP", abp), ("PPG", ppg), ("ECG", ecg)]
        t = np.arange(len(abp)) / fs
        for (name, sig), ax in zip(signals, axes):
            ax.plot(t, sig, lw=2)
            min_v, max_v = np.nanmin(sig), np.nanmax(sig)
            ax.fill_between(t, min_v, max_v, where=is_good, color="green", alpha=0.1)
            ax.fill_between(t, min_v, max_v, where=~is_good, color="red", alpha=0.1)
            ax.set_title(name, fontsize=22)
        percent_good = is_good.sum() / len(is_good) * 100
        fig.suptitle(f"Percent of good segments: {round(percent_good, 2)}%", fontsize=26)
    return good_segments


def find_valid_segments_for_single_measurement(measurement_path, fs, sample_len_sec, max_n_same, sig_names, plot=False):
    """Find valid segments for given measurement path using validate_recording func."""
    data = load_data_to_array(measurement_path, sig_names=sig_names)
    abp, ppg, ecg = data.T
    valid_segments = find_valid_segments(
        abp, ppg, ecg, fs=fs, max_n_same=max_n_same, sample_len_sec=sample_len_sec, plot=plot
    )
    valid_segments_lines = [f"{measurement_path}/{start}-{end}" for start, end in valid_segments]
    append_txt_to_file(LOGS_PATH / "valid_segments.txt", valid_segments_lines)
    append_txt_to_file(LOGS_PATH / "segmented_measurements.txt", measurement_path)
    log.info(f"{measurement_path}: Found {len(valid_segments)} valid segments.")
    return valid_segments


def remove_single_measurement(measurement_path):
    log.info(f"{measurement_path}: Performing measurement (.dat and .hea files) removal.")
    path = str(MIMIC_DB_PATH / measurement_path)
    os.remove(f"{path}.hea")
    os.remove(f"{path}.dat")
    append_txt_to_file(LOGS_PATH / "removed_measurements.txt", measurement_path)


def cut_segments_into_samples(segments, sample_len_sec, fs, shuffle=True):
    """Split bigger segments bounds into smaller samples bounds"""
    sample_len_samples = sample_len_sec * fs
    samples_bounds = []
    for start, end in segments:
        segment_start = start
        segment_end = start + sample_len_samples
        while segment_end < end:
            samples_bounds.append((segment_start, segment_end))
            segment_start, segment_end = segment_end, segment_end + sample_len_samples
    if shuffle:
        random.shuffle(samples_bounds)
    return samples_bounds


def load_segment_to_array(measurement_path, start, end, sig_names):
    """Load data to DataFrame for specific bounds using wfdb library"""
    return load_data_to_array(measurement_path, sig_names=sig_names)[start:end]


def save_sample_to_numpy(sample_num, measurement_path, sample_start_idx, sample_end_idx, sig_names):
    """Save sample to csv file."""
    subject_prefix, subject, measurement = measurement_path.split("/")
    data = load_segment_to_array(measurement_path, sample_start_idx, sample_end_idx, sig_names=sig_names)
    np.save(str(RAW_DATASET_PATH / subject_prefix / subject / f"{subject}_{sample_num}.npy"), data)
    append_txt_to_file(
        SAMPLE_SEGMENTS_FILE_PATH, f"{subject}_{sample_num},{measurement_path},{sample_start_idx},{sample_end_idx}"
    )


def cut_subject_all_segments_into_samples(
    subject, subject_segments, fs, sample_len_sec, max_samples_per_subject, sig_names
):
    subject_path = RAW_DATASET_PATH / subject
    subject_path.mkdir(parents=True, exist_ok=True)

    subject_measurements = [path.split("/")[2] for path in subject_segments.keys()]
    n_samples = 0
    measurement_idx = 0

    cut_measurements_segments = {}
    for path, segments in subject_segments.items():
        segments = np.array(segments)
        cut_segments = cut_segments_into_samples(segments, fs=fs, sample_len_sec=sample_len_sec, shuffle=True)
        cut_measurements_segments[path] = cut_segments
    subject_data = []
    while n_samples < max_samples_per_subject:
        current_measurement = subject_measurements[measurement_idx]
        measurement_path = f"{subject}/{current_measurement}"
        possible_segments_bounds = cut_measurements_segments[measurement_path]
        try:
            sample_start_idx, sample_end_idx = possible_segments_bounds.pop()
        except IndexError:
            break  # no more samples
        sample_path = f"{measurement_path}/{sample_start_idx}-{sample_end_idx}"
        if sample_path not in subject_data:
            subject_data.append({"path": measurement_path, "start": sample_start_idx, "end": sample_end_idx})
            save_sample_to_numpy(n_samples, measurement_path, sample_start_idx, sample_end_idx, sig_names=sig_names)
        n_samples += 1
        measurement_idx += 1
        if measurement_idx > len(subject_measurements) - 1:
            measurement_idx = 0
    log.info(f"{subject}: {len(subject_data)} samples for that subject")
    return {"subject": subject, "samples": subject_data}


def segment_and_save_for_subject(subject, fs, sample_len_sec, max_n_same, sig_names, max_samples_per_subject):
    sample_len_samples = sample_len_sec * fs
    measurements_paths = get_measurements_paths_for_single_subject(subject)
    if not measurements_paths:
        log.warning(f"{subject}: No measurements")
        return {subject: None}
    signals_availability = pd.DataFrame(
        [check_signals_availability_for_single_measurement(path, sig_names=sig_names) for path in measurements_paths]
    )

    valid_measurements = signals_availability.query(
        f"ABP == True and PLETH == True and II == True and n_samples >= {sample_len_samples}"
    )
    if valid_measurements.empty:
        log.warning(
            f"{subject}: Required signals not available for subject {subject} or to short. Returing empty dict."
        )
        return {path: None for path in signals_availability["path"].values}
    else:
        log.info(
            f"{subject}: {len(valid_measurements)} of {len(measurements_paths)} measurements are valid in terms of signals availability"
        )

    valid_measurements_paths = valid_measurements["path"].values

    valid_segmented_measurements = {}
    for path in valid_measurements_paths:
        download_single_measurement(path)
        valid_segments = find_valid_segments_for_single_measurement(
            path, fs=fs, sample_len_sec=sample_len_sec, max_n_same=max_n_same, sig_names=sig_names
        )
        if len(valid_segments) == 0:
            remove_single_measurement(path)
        valid_segmented_measurements[path] = valid_segments

    filtered_segmented_measurements = {
        path: segments for path, segments in valid_segmented_measurements.items() if len(segments) > 0
    }
    if filtered_segmented_measurements:
        params = dict(
            fs=fs, sample_len_sec=sample_len_sec, max_samples_per_subject=max_samples_per_subject, sig_names=sig_names
        )
        cut_subject_all_segments_into_samples(subject, filtered_segmented_measurements, **params)
    else:
        log.warning(f"{subject}: No samples for that subject")
    return valid_segmented_measurements


def save_segments_info_to_file(valid_segmented_measurements):
    lines = []
    for measurement_path, segments in valid_segmented_measurements.items():
        path_split = measurement_path.split("/")
        if len(path_split) == 2:
            subject_prefix, subject = path_split
            txt = [
                f"{subject_prefix},{subject},{np.nan},{np.nan},{np.nan},{np.nan}"
            ]  # No measurements for that subject
        else:
            subject_prefix, subject, measurement = path_split
            if segments is None:
                txt = [
                    f"{subject_prefix},{subject},{measurement},{False},{np.nan},{np.nan}"
                ]  # No measurements with all signals available
                lines.append(txt)
            elif len(segments) == 0:
                txt = [
                    f"{subject_prefix},{subject},{measurement},{True},{np.nan},{np.nan}"
                ]  # No measurements with valid segments
            else:
                txt = []
                for start_idx, end_idx in segments:
                    txt.append(
                        f"{subject_prefix},{subject},{measurement},{True},{int(start_idx)},{int(end_idx)}"
                    )  # Valid segments
        lines.extend(txt)
    append_txt_to_file(SEGMENTS_FILE_PATH, lines)


def segment_and_save_for_all_subjects(subjects, fs, sample_len_sec, max_n_same, sig_names, max_samples_per_subject):
    subjects_segments = []
    with multiprocessing.Pool(initializer=set_global_session, processes=8) as pool:
        segment_fn = partial(
            segment_and_save_for_subject,
            fs=fs,
            sample_len_sec=sample_len_sec,
            max_n_same=max_n_same,
            sig_names=sig_names,
            max_samples_per_subject=max_samples_per_subject,
        )
        for valid_segmented_measurements in tqdm(
            pool.imap_unordered(segment_fn, subjects), total=len(subjects), desc="Subjects"
        ):
            subjects_segments.append(valid_segmented_measurements)
            save_segments_info_to_file(valid_segmented_measurements)
    return subjects_segments


def run_pipeline(fs, sample_len_sec, max_n_same, sig_names, max_samples_per_subject):
    log.info("Performing MIMIC raw data creation")
    all_subjects = requests.get(f"{MIMIC_URL}/RECORDS").text.split("/\n")[:-1]
    try:
        segments_info = pd.read_csv(SEGMENTS_FILE_PATH)
        segmented_subjects = np.unique(
            ["/".join(prefix_subject) for prefix_subject in segments_info[["subject_prefix", "subject"]].values]
        )
    except FileNotFoundError as e:
        log.warning(f"{e}. No segmented subjects")
        segmented_subjects = []
    subjects_to_segment = list(sorted(set(all_subjects).difference(set(segmented_subjects))))
    log.info(f"{len(subjects_to_segment)} subjects left (out of {len(all_subjects)})")
    params = dict(
        fs=fs,
        sample_len_sec=sample_len_sec,
        max_n_same=max_n_same,
        sig_names=sig_names,
        max_samples_per_subject=max_samples_per_subject,
    )
    segment_and_save_for_all_subjects(subjects_to_segment, **params)


def prepare_txt_files():
    try:
        with open(SAMPLE_SEGMENTS_FILE_PATH, "r") as myfile:
            header = myfile.read().split("\n")[0]
    except FileNotFoundError as e:
        log.warning(
            f"{SAMPLE_SEGMENTS_FILE_PATH} not found. seting header and appending it to {SAMPLE_SEGMENTS_FILE_PATH} file"
        )
        header = ""
    if header != "sample_id,measurement_path,sample_start_idx,sample_end_idx":
        append_txt_to_file(SAMPLE_SEGMENTS_FILE_PATH, "sample_id,measurement_path,sample_start_idx,sample_end_idx")

    try:
        with open(SEGMENTS_FILE_PATH, "r") as myfile:
            header = myfile.read().split("\n")[0]
    except FileNotFoundError as e:
        log.warning(f"{SEGMENTS_FILE_PATH} not found. Setting header and appending it to {SEGMENTS_FILE_PATH} file")
        header = ""
    if header != "subject_prefix,subject,measurement,signals_availability,start_idx,end_idx":
        append_txt_to_file(
            SEGMENTS_FILE_PATH, "subject_prefix,subject,measurement,signals_availability,start_idx,end_idx"
        )


def get_all_samples_paths():
    all_sample_paths = []
    prefixes = glob.glob(str(RAW_DATASET_PATH / "*"))
    for prefix in prefixes:
        subjects = glob.glob(f"{prefix}/*")
        for subject in subjects:
            samples = glob.glob(f"{subject}/*.npy")
            for sample in samples:
                prefix, subject, sample_id = sample.split("/")[-3:]
                all_sample_paths.append(
                    {"subject": subject, "sample_id": sample_id, "path": f"{prefix}/{subject}/{sample_id}"}
                )
    return pd.DataFrame(all_sample_paths)


def concat_samples(paths, save_path=None):
    numpy_files = [RAW_DATASET_PATH / path for path in paths]
    return np.stack([np.load(f) for f in numpy_files])


def create_raw_tensors_dataset(train_size, val_size, test_size, max_samples_per_subject, random_state):
    info = get_all_samples_paths()
    info["sample_num"] = [int(sample_id.split("_")[1].split(".")[0]) for sample_id in info["sample_id"].values]
    info = info.query(f"sample_num < {max_samples_per_subject}")
    splits_info = create_train_val_test_split_info(
        groups=info["subject"].values,
        info=info,
        train_size=train_size,
        val_size=val_size,
        test_size=test_size,
        random_state=random_state,
    )
    splits_info.to_csv(SPLIT_INFO_PATH)
    for split in splits_info["split"].unique():
        current_split_info = splits_info.query(f"split == '{split}'")
        all_data = concat_samples(paths=current_split_info["path"].values)  # shape [batch, n_samples, n_signals]
        all_data = torch.from_numpy(all_data)
        # signals order: ABP, PPG, ECG
        targets = all_data[..., 0]  # ABP
        data = all_data[..., 1:]  # PPG and ECG
        print(all_data.shape, data.shape, targets.shape)
        torch.save(data, RAW_TENSORS_DATA_PATH / f"{split}_data.pt")
        torch.save(targets, TARGETS_PATH / f"{split}_targets.pt")


@hydra.main(version_base=None, config_path="../../configs/data", config_name="raw")
def main(cfg: DictConfig):
    cfg = cfg.mimic

    log.info(cfg)

    for path in [MIMIC_PATH, RAW_DATASET_PATH, LOGS_PATH, RAW_TENSORS_DATA_PATH, TARGETS_PATH]:
        path.mkdir(parents=True, exist_ok=True)

    prepare_txt_files()
    if cfg.download:
        log.info("Downloading and segmenting MIMIC dataset")
        run_pipeline(
            fs=cfg.fs,
            sample_len_sec=cfg.sample_len_sec,
            max_n_same=cfg.max_n_same,
            sig_names=cfg.sig_names,
            max_samples_per_subject=cfg.max_samples_per_subject,
        )

    if cfg.create_splits:
        log.info("Creating train/val/test tensors")
        create_raw_tensors_dataset(
            train_size=cfg.split.train_size,
            val_size=cfg.split.val_size,
            test_size=cfg.split.test_size,
            max_samples_per_subject=cfg.split.max_samples_per_subject_for_split,
            random_state=cfg.split.random_state,
        )


if __name__ == "__main__":
    main()
