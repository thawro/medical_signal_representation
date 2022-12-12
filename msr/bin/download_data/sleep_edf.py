import logging

import hydra
from omegaconf import DictConfig, OmegaConf

log = logging.getLogger(__name__)
log.setLevel(logging.INFO)
from msr.data.download.sleep_edf import (
    DATASET_PATH,
    RAW_TENSORS_PATH,
    SPLIT_INFO_PATH,
    ZIP_FILE_URL,
    create_raw_csv_dataset,
    create_raw_tensors_dataset,
    get_sleep_edf_raw_data_info,
)
from msr.data.utils import create_train_val_test_split_info
from msr.utils import CONFIG_PATH, download_zip_and_extract, print_config_tree


@hydra.main(version_base=None, config_path=CONFIG_PATH / "download_data", config_name="sleep_edf")
def main(cfg: DictConfig):
    log.info("Downloading Sleep-EDF data and saving raw tensors")
    print_config_tree(cfg, keys="all")

    if cfg.download:
        download_zip_and_extract(zip_file_url=ZIP_FILE_URL, dest_path=DATASET_PATH)

    if cfg.create_raw_csv:
        get_sleep_edf_raw_data_info()
        create_raw_csv_dataset(
            sig_names=cfg.sig_names,
            sample_len_sec=cfg.sample_len_sec,
            verbose=cfg.verbose,
        )

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
    log.info("Data downloaded and saved as tensors")
    OmegaConf.save(cfg, RAW_TENSORS_PATH / "config.yaml")
    log.info("Config file saved in representations directory")
    log.info("Finished")


if __name__ == "__main__":
    main()
