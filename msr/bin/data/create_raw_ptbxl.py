import logging

import hydra
from omegaconf import DictConfig

from msr.data.raw.ptbxl import DATASET_PATH, ZIP_FILE_URL, create_raw_tensors_dataset
from msr.utils import download_zip_and_extract

log = logging.getLogger(__name__)
log.setLevel(logging.INFO)


@hydra.main(version_base=None, config_path="../../configs/data", config_name="raw_ptbxl")
def main(cfg: DictConfig):
    log.info(cfg)

    if cfg.download:
        download_zip_and_extract(zip_file_url=ZIP_FILE_URL, dest_path=DATASET_PATH)

    if cfg.create_splits:
        create_raw_tensors_dataset(fs=cfg.fs, target=cfg.target, encode_targets=cfg.encode_targets)


if __name__ == "__main__":
    main()
