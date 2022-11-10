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
from msr.utils import download_zip_and_extract, print_config_tree


@hydra.main(version_base=None, config_path="../../configs/data", config_name=f"sleep_edf")
def main(cfg: DictConfig):
    log.info("Creating Sleep-EDF raw tensors dataset")
    print_config_tree(cfg, keys=["raw_data", "create_raw_tensors"])

    if cfg.create_raw_tensors.download:
        download_zip_and_extract(zip_file_url=ZIP_FILE_URL, dest_path=DATASET_PATH)

    if cfg.create_raw_tensors.create_raw_csv:
        get_sleep_edf_raw_data_info()
        create_raw_csv_dataset(
            sig_names=cfg.create_raw_tensors.sig_names,
            sample_len_sec=cfg.create_raw_tensors.sample_len_sec,
            verbose=cfg.create_raw_tensors.verbose,
        )

    if cfg.create_raw_tensors.create_splits:
        info = get_sleep_edf_raw_data_info()
        splits_info = create_train_val_test_split_info(
            groups=info["subject"].values,
            info=info,
            train_size=cfg.create_raw_tensors.split.train_size,
            val_size=cfg.create_raw_tensors.split.val_size,
            test_size=cfg.create_raw_tensors.split.test_size,
            random_state=cfg.create_raw_tensors.split.random_state,
        )
        splits_info.to_csv(SPLIT_INFO_PATH, index=False)
        create_raw_tensors_dataset()
    log.info("Finished.")


if __name__ == "__main__":
    main()
