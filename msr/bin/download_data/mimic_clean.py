import logging

import hydra
from omegaconf import DictConfig, OmegaConf

from msr.data.download.mimic_clean import (
    RAW_TENSORS_PATH,
    prepare_mimic_clean_data,
    save_data_and_targets_to_files,
)
from msr.data.utils import create_train_val_test_split_info
from msr.utils import CONFIG_PATH, print_config_tree

log = logging.getLogger(__name__)
log.setLevel(logging.INFO)


@hydra.main(version_base=None, config_path=CONFIG_PATH / "download_data", config_name="mimic_clean")
def main(cfg: DictConfig):
    log.info("Downloading MIMIC data and saving raw tensors")
    print_config_tree(cfg, keys="all")

    data, info = prepare_mimic_clean_data(
        max_samples_per_subject=cfg.max_samples_per_subject, n_samples=cfg.n_samples, seed=cfg.seed
    )

    splits_info = create_train_val_test_split_info(
        groups=info["subject_id"].values,
        info=info,
        train_size=cfg.train_size,
        val_size=cfg.val_size,
        test_size=cfg.test_size,
        random_state=cfg.seed,
    )

    save_data_and_targets_to_files(data, splits_info)

    log.info("Data downloaded and saved as tensors")
    OmegaConf.save(cfg, RAW_TENSORS_PATH / "config.yaml")
    log.info("Config file saved in representations directory")
    log.info("Finished")


if __name__ == "__main__":
    main()
