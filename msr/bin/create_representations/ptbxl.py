import logging

import hydra
from omegaconf import DictConfig, OmegaConf

from msr.data.representation.ptbxl import (
    DATASET_PATH,
    create_ptbxl_representations_dataset,
)
from msr.utils import CONFIG_PATH, print_config_tree

log = logging.getLogger(__name__)
log.setLevel(logging.INFO)


@hydra.main(version_base=None, config_path=CONFIG_PATH / "create_representations", config_name="ptbxl")
def main(cfg: DictConfig):
    log.info("Creating PTB-XL representations dataset")

    print_config_tree(cfg, keys="all")

    create_ptbxl_representations_dataset(
        representation_types=cfg.representation_types,
        beats_params=cfg.beats_params,
        n_beats=cfg.n_beats,
        agg_beat_params=cfg.agg_beat_params,
        windows_params=cfg.windows_params,
        fs=cfg.raw_data.fs,
    )
    REPRESENTATIONS_PATH = DATASET_PATH / f"representations_{cfg.raw_data.fs}"
    log.info("Representasions dataset creation finished")
    OmegaConf.save(cfg, REPRESENTATIONS_PATH / "config.yaml")
    log.info("Config file saved in representations directory")
    log.info("Finished")


if __name__ == "__main__":
    main()
