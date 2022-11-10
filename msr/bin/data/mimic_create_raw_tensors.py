import logging

import hydra
from omegaconf import DictConfig

log = logging.getLogger(__name__)
log.setLevel(logging.INFO)

from msr.data.raw.mimic import (
    create_raw_tensors_dataset,
    download_validate_and_segment,
    prepare_txt_files,
)
from msr.utils import print_config_tree


@hydra.main(version_base=None, config_path="../../configs/data", config_name="mimic")
def main(cfg: DictConfig):
    log.info("Creating MIMIC raw tensors dataset")
    print_config_tree(cfg, keys=["raw_data", "create_raw_tensors"])

    if cfg.create_raw_tensors.download:
        prepare_txt_files()
        download_validate_and_segment(
            sample_len_samples=int(cfg.create_raw_tensors.sample_len_sec * cfg.raw_data.fs),
            max_n_same=cfg.create_raw_tensors.max_n_same,
            sig_names=cfg.create_raw_tensors.sig_names,
            max_samples_per_subject=cfg.create_raw_tensors.max_samples_per_subject,
        )

    if cfg.create_raw_tensors.create_splits:
        create_raw_tensors_dataset(
            train_size=cfg.create_raw_tensors.split.train_size,
            val_size=cfg.create_raw_tensors.split.val_size,
            test_size=cfg.create_raw_tensors.split.test_size,
            max_samples_per_subject=cfg.create_raw_tensors.split.max_samples_per_subject_for_split,
            random_state=cfg.create_raw_tensors.split.random_state,
            fs=cfg.raw_data.fs,
            targets=cfg.create_raw_tensors.split.targets,
        )
    log.info("Finished.")


if __name__ == "__main__":
    main()
