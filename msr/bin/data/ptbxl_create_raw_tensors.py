import logging

import hydra
from omegaconf import DictConfig

from msr.data.raw.ptbxl import DATASET_PATH, ZIP_FILE_URL, create_raw_tensors_dataset
from msr.utils import download_zip_and_extract, print_config_tree

log = logging.getLogger(__name__)
log.setLevel(logging.INFO)


@hydra.main(version_base=None, config_path="../../configs/data", config_name="ptbxl")
def main(cfg: DictConfig):
    log.info("Creating PTB-XL raw tensors dataset")
    print_config_tree(cfg, keys=["raw_data", "create_raw_tensors"])

    if cfg.create_raw_tensors.download:
        download_zip_and_extract(zip_file_url=ZIP_FILE_URL, dest_path=DATASET_PATH)

    if cfg.create_raw_tensors.create_splits:
        create_raw_tensors_dataset(
            fs=cfg.raw_data.fs, target=cfg.raw_data.target, encode_targets=cfg.create_raw_tensors.encode_targets
        )
    log.info("Finished.")


if __name__ == "__main__":
    main()
