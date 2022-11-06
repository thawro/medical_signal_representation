import logging

import hydra
from omegaconf import DictConfig

log = logging.getLogger(__name__)
log.setLevel(logging.INFO)
from msr.data.raw.sleep_edf import (
    DATASET_PATH,
    SPLIT_INFO_PATH,
    ZIP_FILE_URL,
    create_raw_csv_dataset,
    create_raw_tensors_dataset,
    get_sleep_edf_raw_data_info,
)
from msr.data.utils import create_train_val_test_split_info
from msr.utils import download_zip_and_extract


@hydra.main(version_base=None, config_path="../../configs/data", config_name=f"raw_sleep_edf")
def main(cfg: DictConfig):
    log.info(cfg)

    if cfg.download:
        download_zip_and_extract(zip_file_url=ZIP_FILE_URL, dest_path=DATASET_PATH)

    if cfg.create_raw_csv:
        get_sleep_edf_raw_data_info()
        create_raw_csv_dataset(sig_names=cfg.sig_names, sample_len_sec=cfg.sample_len_sec, verbose=cfg.verbose)

    if cfg.create_splits:
        info = get_sleep_edf_raw_data_info()
        splits_info = create_train_val_test_split_info(
            groups=info["subject"].values,
            info=info,
            train_size=cfg.split.train_size,
            val_size=cfg.split.val_size,
            test_size=cfg.split.test_size,
            random_state=cfg.split.random_state,
        )
        splits_info.to_csv(SPLIT_INFO_PATH, index=False)
        create_raw_tensors_dataset()


if __name__ == "__main__":
    main()
