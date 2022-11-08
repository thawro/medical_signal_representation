import logging

import hydra
from omegaconf import DictConfig

from msr.data.representation.ptbxl import create_ptbxl_representations_dataset
from msr.utils import print_config_tree

log = logging.getLogger(__name__)
log.setLevel(logging.INFO)


@hydra.main(version_base=None, config_path="../../configs/data", config_name="ptbxl")
def main(cfg: DictConfig):
    log.info("Creating PTB-XL representations dataset")

    print_config_tree(cfg, keys=["raw_data", "create_representations"])

    create_ptbxl_representations_dataset(
        representation_types=cfg.create_representations.representation_types,
        beats_params=cfg.create_representations.beats_params,
        n_beats=cfg.create_representations.n_beats,
        agg_beat_params=cfg.create_representations.agg_beat_params,
        windows_params=cfg.create_representations.windows_params,
        fs=cfg.raw_data.fs,
    )


if __name__ == "__main__":
    main()
