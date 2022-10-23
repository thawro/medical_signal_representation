from signals.representation_extractor import (
    PeriodicRepresentationExtractor,
    RepresentationExtractor,
)


def get_representations(measurement) -> Dict[str, torch.Tensor]:
    """Get all types of representations (returned by ECGSignal objects).

    Args:
        channels_data (torch.Tensor): ECG channels data of shape `[TODO]`, where TODO.
        fs (float): Sampling rate. Defaults to 100.

    Returns:
        Dict[str, torch.Tensor]: Dict with representations names as keys and `torch.Tensor` objects as values.
    """
    multi_ecg = create_multichannel_ecg(channels_data.T.numpy(), fs)

    n_beats = 10
    multi_ecg.get_beats(source_channel=2)
    whole_signal_waveforms = multi_ecg.get_waveform_representation()
    whole_signal_features = multi_ecg.get_whole_signal_features_representation()
    per_beat_waveforms = multi_ecg.get_per_beat_waveform_representation(n_beats=n_beats)
    per_beat_features = multi_ecg.get_per_beat_features_representation(n_beats=n_beats)
    # whole_signal_embeddings = multi_ecg.get_whole_signal_embeddings_representation()

    actual_beats = per_beat_waveforms.shape[1]
    n_more_beats = n_beats - actual_beats
    # print(f"{actual_beats} beats in signal. Padding {n_more_beats} beats")
    per_beat_waveforms = np.pad(per_beat_waveforms, ((0, 0), (0, n_more_beats), (0, 0)))  # padding beats with zeros
    per_beat_features = np.pad(per_beat_features, ((0, 0), (0, n_more_beats), (0, 0)))  # padding beats with zeros

    return {
        "whole_signal_waveforms": whole_signal_waveforms,
        "whole_signal_features": whole_signal_features,
        "per_beat_waveforms": per_beat_waveforms,
        "per_beat_features": per_beat_features,
        # 'whole_signal_embeddings': whole_signal_embeddings,
    }


def create_representations_dataset(
    splits: List[str] = ["train", "val", "test"], fs: float = 100, use_multiprocessing: bool = True
):
    """Create and save data files (`.pt`) for all representations.

    Args:
        splits (list): Split types to create representations data for. Defaults to ['train', 'val', 'test'].
        fs (float): Sampling frequency of signals. Defaults to 100.
        use_multiprocessing (bool): Whether to use `joblib` library for multiprocessing
            or run everything in simple `for` loop. Defaults to `True`.
    """

    REPRESENTATIONS_DATA_PATH = PTBXL_PATH / f"representations_{fs}"
    REPRESENTATIONS_DATA_PATH.mkdir(parents=True, exist_ok=True)

    for split in tqdm(splits):
        data = torch.load(RAW_TENSORS_DATA_PATH / f"records{fs}/{split}_data.pt")
        if use_multiprocessing:
            representations = Parallel(n_jobs=-1)(delayed(get_representations)(channels, fs) for channels in tqdm(data))
        else:
            representations = [get_representations(channels) for channels in tqdm(data)]
        representation_names = representations[0].keys()
        representations = {
            name: torch.tensor(np.array([rep[name] for rep in representations])) for name in representation_names
        }
        for name, representation_data in representations.items():
            path = REPRESENTATIONS_DATA_PATH / name / f"{split}_data.pt"
            torch.save(representation_data, path)
