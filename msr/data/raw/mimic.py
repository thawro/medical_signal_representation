import glob
import logging
import multiprocessing
import os
from functools import partial
from typing import Dict, List, Literal, Tuple

import matplotlib.pyplot as plt
import neurokit2 as nk
import numpy as np
import pandas as pd
import requests
import torch
import wfdb
from requests.adapters import HTTPAdapter
from requests.packages.urllib3.util.retry import Retry
from tqdm.auto import tqdm

from msr.data.raw.utils import cut_segments_into_samples, validate_signal
from msr.data.utils import create_train_val_test_split_info
from msr.signals.ecg import ECGSignal, check_ecg_polarity, find_intervals_using_hr
from msr.signals.utils import resample
from msr.utils import DATA_PATH, append_txt_to_file, find_true_intervals, load_tensor

log = logging.getLogger(__name__)
log.setLevel(logging.INFO)

DATASET_NAME = "mimic"
MIMIC_DB_NAME = "mimic3wdb-matched"  # specify type of mimic databes (mimic3wdb or mimic3wdb-matched)
MIMIC_URL = f"https://physionet.org/files/{MIMIC_DB_NAME}/1.0"
DATASET_PATH = DATA_PATH / DATASET_NAME
MIMIC_DB_PATH = DATASET_PATH / "db"  # raw data in .dat and .hea formats
LOGS_PATH = DATASET_PATH / "logs"
FS = 125

SEGMENTS_FILE_PATH = LOGS_PATH / "segments_info.txt"  # file for segments info
SAMPLE_SEGMENTS_FILE_PATH = LOGS_PATH / "sample_segments_info.txt"  # filename for samples segments info
SPLIT_INFO_PATH = LOGS_PATH / "split_info.csv"
VALID_SEGMENTS_PATH = LOGS_PATH / "valid_segments.txt"

RAW_DATASET_PATH = DATASET_PATH / "raw"
RAW_TENSORS_PATH = DATASET_PATH / "raw_tensors"
RAW_TENSORS_DATA_PATH = RAW_TENSORS_PATH / "data"
TARGETS_PATH = RAW_TENSORS_PATH / "targets"
for path in [RAW_DATASET_PATH, LOGS_PATH, RAW_TENSORS_DATA_PATH, TARGETS_PATH]:
    path.mkdir(parents=True, exist_ok=True)
SESSION = None


def set_global_session():
    """Set global session used for files downloading"""
    global SESSION
    if not SESSION:
        SESSION = requests.Session()
        retry = Retry(connect=100, total=100, backoff_factor=0.1)
        adapter = HTTPAdapter(max_retries=retry)
        SESSION.mount("http://", adapter)
        SESSION.mount("https://", adapter)


def get_measurements_paths(subject: str) -> List[str]:
    """Return measurements paths for single subject using RECORDS file in physionet mimic3db database.

    Args:
        subject (str): Subject code.

    Returns:
        List[str]: Measurement paths.
    """
    url = f"{MIMIC_URL}/{subject}/RECORDS"
    with SESSION.get(url) as response:
        measurements = response.text.split("\n")[:-1]
        measurements = [f"{subject}/{measurement}" for measurement in measurements if "-" not in measurement]
        return measurements


def check_signals_availability(measurement_path: str, sig_names: List[str]) -> Dict:
    """Check if all desired signals are available (in the header file).

    Args:
        measurement_path (str): Measurement path in format <prefix>/<subject>/<measurement>.
        sig_names (List[str]): Required signals names.

    Returns:
        Dict: Dictionary with measurement path, signals availability info and number of samples info.
    """
    chunksize = 1024 * 1024
    directory, subject, measurement = measurement_path.split("/")
    url_hea = f"{MIMIC_URL}/{directory}/{subject}/{measurement}.hea"
    tmp_path = DATASET_PATH / "tmp"
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
    return {
        "path": measurement_path,
        **{sig_name: sig_name in header.sig_name for sig_name in sig_names},
        "n_samples": header.sig_len,
    }


def download_measurement(measurement_path: str):
    """Download measurement from physionet database.

    Args:
        measurement_path (str): Measurement path in format <prefix>/<subject>/<measurement>.
    """

    def download_file(url: str, filename: str, chunksize=1024 * 1024):
        """Download file with previously set session"""
        with SESSION.get(url, stream=True) as response:
            response.raise_for_status()
            with open(filename, "wb") as f:
                for chunk in response.iter_content(chunk_size=chunksize):
                    f.write(chunk)

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
    log.info(f"{measurement_path}: downloaded")


def remove_measurement(measurement_path: str):
    """Remove measurements .hea and .dat files

    Args:
        measurement_path (str): Measurement path in format <prefix>/<subject>/<measurement>.
    """
    log.info(f"{measurement_path}: Removing .dat and .hea files.")
    path = str(MIMIC_DB_PATH / measurement_path)
    os.remove(f"{path}.hea")
    os.remove(f"{path}.dat")


def load_raw_measurement_to_array(
    measurement_path: str, sig_names: List[str], start_idx: int = 0, end_idx: int = -1
) -> np.ndarray:
    """Load data to numpy.ndarray using wfdb library

    Args:
        measurement_path (str): Measurement path in format <prefix>/<subject>/<measurement>.
        sig_names (List[str]): Required signals names.
        start_idx (int, optional): Starting index of measurements slice. Defaults to 0.
        end_idx (int, optional): Ending index of measurements slice. If -1, then length of measurement is used. Defaults to -1.

    Returns:
        np.ndarray: Numpy array with signals data.
    """
    path = str(MIMIC_DB_PATH / measurement_path)
    try:
        data, info = wfdb.rdsamp(path)
    except ValueError as e:
        log.error(f"{measurement_path}: {e}. mimic db file was corrupted. Downloading it again")
        with multiprocessing.Pool(initializer=set_global_session, processes=1) as pool:
            pool.imap_unordered(download_measurement, [measurement_path])
        data, info = wfdb.rdsamp(path)
    sample_sig_names = info["sig_name"]
    sig_idxs = [sample_sig_names.index(sig_name) for sig_name in sig_names]
    end_idx = len(data) if end_idx < start_idx else end_idx
    return data[start_idx:end_idx, sig_idxs]  # data of shape [N, C], where C are signals from sig_names


def validate_recording(
    abp: np.ndarray, ppg: np.ndarray, ecg: np.ndarray, sample_len_samples: int, max_n_same: int, plot: bool = False
) -> np.ndarray:
    """Check if recording is valid.
    All signals (ABP, PPG and ECG) are checked for values bounds, nans and repeated values.

    Args:
        abp (np.ndarray): ABP signal.
        ppg (np.ndarray): PPG (PLETH) signal.
        ecg (np.ndarray): ECG (lead II) signal.
        sample_len_samples (int): Minimum accepted signal length (in samples).
        max_n_same (int): Maximum number of repeated samples.
        plot (bool): Whether to plot validation process. Defaults to False.

    Returns:
        np.ndarray: True indicates samples which are good for all validation steps for all signals.
    """
    valid_sig_len = len(abp) >= sample_len_samples

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
        t = np.arange(len(abp))
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


def find_valid_segments(
    measurement_path: str, sample_len_samples: int, max_n_same: int, sig_names: List[str], plot: bool = False
) -> List[Tuple[int, int]]:
    """Find valid segments for given measurement.

    Args:
        measurement_path (str): Measurement path in format <prefix>/<subject>/<measurement>.
        sample_len_samples (int): Minimum accepted signal length (in samples).
        max_n_same (int): Maximum number of repeated samples.
        sig_names (List[str]): Required signals names.
        plot (bool): Whether to plot validation process. Defaults to False.

    Returns:
        List[Tuple[int, int]]: List with valid segments intervals.
    """
    data = load_raw_measurement_to_array(measurement_path, sig_names=sig_names)
    abp, ppg, ecg = data.T
    is_good = validate_recording(abp, ppg, ecg, sample_len_samples, max_n_same, plot=plot)
    segments = find_true_intervals(is_good)
    valid_segments = [segment for segment in segments if (segment[1] - segment[0]) >= sample_len_samples]
    if plot:
        is_good = np.zeros_like(abp) == 1
        for start, end in valid_segments:
            is_good[start:end] = True
        fig, axes = plt.subplots(3, 1, figsize=(24, 18))
        signals = [("ABP", abp), ("PPG", ppg), ("ECG", ecg)]
        t = np.arange(len(abp))
        for (name, sig), ax in zip(signals, axes):
            ax.plot(t, sig, lw=2)
            min_v, max_v = np.nanmin(sig), np.nanmax(sig)
            ax.fill_between(t, min_v, max_v, where=is_good, color="green", alpha=0.1)
            ax.fill_between(t, min_v, max_v, where=~is_good, color="red", alpha=0.1)
            ax.set_title(name, fontsize=22)
        percent_good = is_good.sum() / len(is_good) * 100
        fig.suptitle(f"Percent of good segments: {round(percent_good, 2)}%", fontsize=26)

    valid_segments_lines = [f"{measurement_path}/{start}-{end}" for start, end in valid_segments]
    append_txt_to_file(VALID_SEGMENTS_PATH, valid_segments_lines)
    log.info(f"{measurement_path}: Found {len(valid_segments)} valid segments.")
    if len(valid_segments) == 0:
        remove_measurement(measurement_path)
    return valid_segments


def save_sample_to_numpy(sample_num: int, measurement_path: str, start_idx: int, end_idx: int, sig_names: List[str]):
    """Save sample to numpy file.

    Args:
        sample_num (int): Sample number.
        measurement_path (str): Measurement path in format <prefix>/<subject>/<measurement>.
        start_idx (int): Starting index of sample.
        end_idx (int): Ending index of sample.
        sig_names (List[str]): Required signals names.
    """
    subject_prefix, subject, measurement = measurement_path.split("/")
    subject_path = RAW_DATASET_PATH / subject_prefix / subject
    subject_path.mkdir(parents=True, exist_ok=True)
    file_path = str(subject_path / f"{subject}_{sample_num}.npy")
    sample_data = load_raw_measurement_to_array(measurement_path, sig_names, start_idx, end_idx)
    np.save(file_path, sample_data)
    log.info(f"{measurement_path}: Saved sample {sample_num}")
    append_txt_to_file(SAMPLE_SEGMENTS_FILE_PATH, f"{subject}_{sample_num},{measurement_path},{start_idx},{end_idx}")


def cut_segments_into_samples_and_save(
    segments: Dict[str, List[Tuple[int, int]]],
    sample_len_samples: int,
    max_samples_per_subject: int,
    sig_names: List[str],
):
    """Cut single subjects segments into samples and save it to numpy files.

    Args:
        segments (Dict[str, List[Tuple[int, int]]]): Dictionary with measurements paths as keys and corresponding segments as values.
        sample_len_samples (int): Minimum accepted signal length (in samples).
        max_samples_per_subject (int): Maximum number of samples to save for one subject.
        sig_names (List[str]): Required signals names.
    """
    measurement_paths = list(segments.keys())
    samples_count = 0
    measurements_count = 0

    cut_measurements_segments = {}
    for path, segments in segments.items():
        segments = np.array(segments)
        cut_segments = cut_segments_into_samples(segments, n_samples=sample_len_samples, shuffle=True)
        cut_measurements_segments[path] = cut_segments
    while samples_count < max_samples_per_subject:
        measurement_path = measurement_paths[measurements_count]
        possible_segments_bounds = cut_measurements_segments[measurement_path]
        try:
            start_idx, end_idx = possible_segments_bounds.pop()
        except IndexError:
            break  # no more samples
        save_sample_to_numpy(samples_count, measurement_path, start_idx, end_idx, sig_names=sig_names)
        samples_count += 1
        measurements_count += 1
        if measurements_count > len(measurement_paths) - 1:
            measurements_count = 0
    log.info(f"Saved {samples_count} samples")


def segment_and_save_for_subject(
    subject: str, sample_len_samples: int, max_n_same: int, max_samples_per_subject: int, sig_names: List[str]
) -> Dict[str, List[Tuple[int, int]]]:
    """Segment subjects measurements and save valid segments to numpy files.

    Args:
        subject (str): Subject code.
        sample_len_samples (int): Minimum accepted signal length (in samples).
        max_n_same (int): Maximum number of repeated samples.
        max_samples_per_subject (int): Maximum number of samples to save for one subject in a split.
        sig_names (List[str]): Required signals names.

    Returns:
        Dict[str, List[Tuple[int, int]]]: Dictionary with measurements paths as keys and corresponding segments as values.
    """
    measurements_paths = get_measurements_paths(subject)
    if not measurements_paths:
        log.warning(f"{subject}: No measurements")
        return {subject: None}
    signals_availability = pd.DataFrame([check_signals_availability(path, sig_names) for path in measurements_paths])

    valid_measurements = signals_availability.query(
        f"ABP == True and PLETH == True and II == True and n_samples >= {sample_len_samples}"
    )
    if valid_measurements.empty:
        log.warning(f"{subject}: Required signals not available for subject {subject} or to short.")
        return {path: None for path in signals_availability["path"].values}
    else:
        log.info(f"{subject}: {len(valid_measurements)}/{len(measurements_paths)} measurements with all signals")

    measurements_paths = valid_measurements["path"].values

    segmented_measurements = {}
    for path in measurements_paths:
        download_measurement(path)
        params = dict(sample_len_samples=sample_len_samples, max_n_same=max_n_same, sig_names=sig_names)
        segmented_measurements[path] = find_valid_segments(path, **params)

    valid_segmented_measurements = {
        path: segments for path, segments in segmented_measurements.items() if len(segments) > 0
    }
    if valid_segmented_measurements:
        params = dict(
            sample_len_samples=sample_len_samples, max_samples_per_subject=max_samples_per_subject, sig_names=sig_names
        )
        cut_segments_into_samples_and_save(valid_segmented_measurements, **params)
    else:
        log.warning(f"{subject}: No samples for that subject")
    return segmented_measurements


def save_segments_info(segmented_measurements: Dict[str, List[Tuple[int, int]]]):
    """Save segments info

    Args:
        segmented_measurements (Dict[str, List[Tuple[int, int]]]): Dictionary with measurements paths as keys and corresponding segments as values.
    """
    lines = []
    for measurement_path, segments in segmented_measurements.items():
        path_split = measurement_path.split("/")
        if len(path_split) == 2:  # No measurements for that subject
            subject_prefix, subject = path_split
            txt = [f"{subject_prefix},{subject},{np.nan},{np.nan},{np.nan},{np.nan}"]
        else:
            subject_prefix, subject, measurement = path_split
            if segments is None:  # No measurements with all signals available
                txt = [f"{subject_prefix},{subject},{measurement},{False},{np.nan},{np.nan}"]
                lines.append(txt)
            elif len(segments) == 0:  # No measurements with valid segments
                txt = [f"{subject_prefix},{subject},{measurement},{True},{np.nan},{np.nan}"]
            else:  # Valid segments
                txt = []
                for start_idx, end_idx in segments:
                    txt.append(f"{subject_prefix},{subject},{measurement},{True},{int(start_idx)},{int(end_idx)}")
        lines.extend(txt)
    append_txt_to_file(SEGMENTS_FILE_PATH, lines)


def segment_and_save_for_all_subjects(
    subjects: List[str], sample_len_samples: int, max_n_same: int, max_samples_per_subject: int, sig_names: List[str]
):
    """Segment each subjects measurements and save valid segments to numpy files.

    Args:
        subjects (List[str]): List of subjects codes.
        sample_len_samples (int): Minimum accepted signal length (in samples).
        max_n_same (int): Maximum number of repeated samples.
        max_samples_per_subject (int): Maximum number of samples to save for one subject.
        sig_names (List[str]): Required signals names.
    """
    with multiprocessing.Pool(initializer=set_global_session, processes=16) as pool:
        segment_fn = partial(
            segment_and_save_for_subject,
            sample_len_samples=sample_len_samples,
            max_n_same=max_n_same,
            sig_names=sig_names,
            max_samples_per_subject=max_samples_per_subject,
        )
        for segmented_measurements in tqdm(
            pool.imap_unordered(segment_fn, subjects), total=len(subjects), desc="Subjects"
        ):
            save_segments_info(segmented_measurements)


def download_validate_and_segment(
    sample_len_samples: int, max_n_same: int, max_samples_per_subject: int, sig_names: List[str]
):
    """Run mimic download -> validate -> segment -> save pipeline.

    Args:
        sample_len_samples (int): Minimum accepted signal length (in samples).
        max_n_same (int): Maximum number of repeated samples.
        max_samples_per_subject (int): Maximum number of samples to save for one subject.
        sig_names (List[str]): Required signals names.
    """
    log.info("Downloading and segmenting MIMIC dataset")
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
        sample_len_samples=sample_len_samples,
        max_n_same=max_n_same,
        sig_names=sig_names,
        max_samples_per_subject=max_samples_per_subject,
    )
    segment_and_save_for_all_subjects(subjects_to_segment, **params)


def prepare_txt_files():
    """Prepare text logs files, i.e. write down needed headers."""
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


def get_all_samples_paths() -> pd.DataFrame:
    """Return all numpy files samples paths in form of pandas.DataFrame"""
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


def find_sbp_and_dbp_idxs_with_rpeaks(
    abp: np.ndarray, ecg: np.ndarray, ecg_fs: float, correct: bool = True
) -> Tuple[np.ndarray, np.ndarray]:
    """Return systolic and diastolic blood pressure indexes (from ABP signal) with use of ECG rpeaks.

    Args:
        abp (np.ndarray): ABP signal from which sbp and dbp indexes are returned.
        ecg (np.ndarray): ECG signal, i.e. source of rpeaks.
        ecg_fs (float): ECG sampling frequency.
        correct (bool, optional): Whether to correct badly extracted indexes. Defaults to True.

    Returns:
        Tuple[np.ndarray, np.ndarray]: Systolic and diastolic indexes.
    """
    cleaned = nk.ecg_clean(ecg, sampling_rate=ecg_fs)
    rpeaks = nk.ecg_peaks(cleaned, sampling_rate=ecg_fs, correct_artifacts=True)[1]["ECG_R_Peaks"]
    if len(rpeaks) == 0:
        ecg_sig = ECGSignal("ecg", ecg, ecg_fs)
        intervals = np.array(find_intervals_using_hr(ecg_sig))
    else:
        beats_n_samples = np.diff(rpeaks)
        min_n_samples = min(beats_n_samples)

        if rpeaks[0] - min_n_samples < 0:
            intervals = np.array([(start, start + min_n_samples) for start in rpeaks])
        else:
            intervals = np.array([(start - min_n_samples, start) for start in rpeaks])

    intervals = intervals[intervals[:, 1] < len(abp)]

    abp_beats = np.array([abp[s:e] for s, e in intervals])

    sbp_idxs = abp_beats.argmax(axis=1) + intervals[:, 0]
    dbp_idxs = abp_beats.argmin(axis=1) + intervals[:, 0]

    if correct:
        mask = sbp_idxs - dbp_idxs > 0
        sbp_idxs, dbp_idxs = sbp_idxs[mask], dbp_idxs[mask]
    return sbp_idxs, dbp_idxs


def get_abp_targets(
    abp: np.ndarray, ecg: np.ndarray, ecg_fs: float, targets: Literal["abp", "sbp_dbp", "sbp_dbp_avg"]
) -> Dict[str, np.ndarray]:
    """Calculate systolic and diastolic targets.

    Args:
        abp (np.ndarray): ABP signal from which sbp and dbp indexes are returned.
        ecg (np.ndarray): ECG signal, i.e. source of rpeaks.
        ecg_fs (float): ECG sampling frequency.
        targets (Literal["abp", "sbp_dbp", "sbp_dbp_avg"]): Which BP targets to return.

    Returns:
        Dict[str, np.ndarray]: Dictionary with targets.
    """
    if isinstance(abp, torch.Tensor):
        abp = abp.numpy()
    if isinstance(ecg, torch.Tensor):
        ecg = ecg.numpy()

    ecg = ecg * check_ecg_polarity(ecg)
    sbp_loc, dbp_loc = find_sbp_and_dbp_idxs_with_rpeaks(abp, ecg, ecg_fs=ecg_fs, correct=False)
    sbp, dbp = abp[sbp_loc], abp[dbp_loc]

    new_time = new_time = np.arange(len(abp))

    bp_values = {}

    if "sbp_dbp" in targets:
        sbp_resampled = resample(sbp_loc, sbp, new_time, kind="nearest")
        dbp_resampled = resample(dbp_loc, dbp, new_time, kind="nearest")
        bp_values["sbp_dbp"] = np.vstack((sbp_resampled, dbp_resampled))

    if "sbp_dbp_avg" in targets:
        bp_values["sbp_dbp_avg"] = np.array([sbp.mean(), dbp.mean()])

    return bp_values


def create_raw_tensors_dataset(
    train_size: float,
    val_size: float,
    test_size: float,
    max_samples_per_subject: int,
    random_state: int,
    fs: float,
    targets: Literal["abp", "sbp_dbp", "sbp_dbp_avg"],
):
    """Create raw tensors dataset, i.e. train/val/test data and targets tensors.

    Args:
        train_size (float): Train ratio of dataset.
        val_size (float): Val ratio of dataset.
        test_size (float): Test ratio of dataset.
        max_samples_per_subject (int): Maximum number of samples to save for one subject.
        random_state (int): Seed for random.
        fs (float): Signals sampling frequency.
        targets (Literal["abp", "sbp_dbp", "sbp_dbp_avg"]): Which BP targets to save.
    """
    log.info("Creating train/val/test tensors")

    def concat_samples(paths: List[str]):
        """Concatenate numpy arrays loaded from paths"""
        numpy_files = [RAW_DATASET_PATH / path for path in paths]
        return np.stack([np.load(f) for f in numpy_files])

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
    splits = splits_info["split"].unique()
    for split in splits:
        current_split_info = splits_info.query(f"split == '{split}'")
        all_data = concat_samples(paths=current_split_info["path"].values)  # shape [batch, n_samples, n_signals]
        all_data = torch.from_numpy(all_data)  # signals order: ABP, PPG, ECG
        abp_data = all_data[..., 0]  # ABP
        data = all_data[..., 1:]  # PPG and ECG
        ecg_data = data[..., 1]
        torch.save(data, RAW_TENSORS_DATA_PATH / f"{split}.pt")

        bp_values = [
            get_abp_targets(abp, ecg, ecg_fs=fs, targets=targets)
            for abp, ecg in tqdm(
                zip(abp_data, ecg_data), total=len(ecg_data), desc=f"Gethering MIMIC targets for {split} split"
            )
        ]
        bp_values = {
            name: torch.tensor(np.array([target[name] for target in bp_values])) for name in bp_values[0].keys()
        }
        if "abp" in targets:
            bp_values["abp"] = abp_data
        for name, target_data in bp_values.items():
            target_path = TARGETS_PATH / name
            target_path.mkdir(parents=True, exist_ok=True)
            torch.save(target_data, target_path / f"{split}.pt")


def load_mimic_raw_tensors_for_split(split):
    data, targets = load_tensor(RAW_TENSORS_DATA_PATH / f"{split}.pt"), load_tensor(TARGETS_PATH / f"{split}.pt")
    return data, targets
