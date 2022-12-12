import logging

import hydra
from omegaconf import DictConfig, OmegaConf

from msr.data.download.ptbxl import (
    DATASET_PATH,
    RAW_TENSORS_PATH,
    ZIP_FILE_URL,
    create_raw_tensors_dataset,
)
from msr.utils import CONFIG_PATH, download_zip_and_extract, print_config_tree

log = logging.getLogger(__name__)
log.setLevel(logging.INFO)


@hydra.main(version_base=None, config_path=CONFIG_PATH / "download_data", config_name="ptbxl")
def main(cfg: DictConfig):
    log.info("Downloading PTB-XL data and saving raw tensors")
    print_config_tree(cfg, keys="all")

    if cfg.download:
        download_zip_and_extract(zip_file_url=ZIP_FILE_URL, dest_path=DATASET_PATH)

    if cfg.create_splits:
        create_raw_tensors_dataset(fs=cfg.raw_data.fs, target=cfg.raw_data.target, encode_targets=cfg.encode_targets)
    log.info("Data downloaded and saved as tensors")
    OmegaConf.save(cfg, RAW_TENSORS_PATH / "config.yaml")
    log.info("Config file saved in representations directory")
    log.info("Finished")


if __name__ == "__main__":
    main()
