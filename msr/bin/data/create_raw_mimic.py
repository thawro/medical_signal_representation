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


@hydra.main(version_base=None, config_path="../../configs/data", config_name="raw_mimic")
def main(cfg: DictConfig):
    log.info(cfg)

    if cfg.download:
        prepare_txt_files()
        download_validate_and_segment(
            sample_len_samples=int(cfg.sample_len_sec * cfg.fs),
            max_n_same=cfg.max_n_same,
            sig_names=cfg.sig_names,
            max_samples_per_subject=cfg.max_samples_per_subject,
        )

    if cfg.create_splits:
        create_raw_tensors_dataset(
            train_size=cfg.split.train_size,
            val_size=cfg.split.val_size,
            test_size=cfg.split.test_size,
            max_samples_per_subject=cfg.split.max_samples_per_subject_for_split,
            random_state=cfg.split.random_state,
            fs=cfg.fs,
            targets=cfg.split.targets,
        )


if __name__ == "__main__":
    main()
