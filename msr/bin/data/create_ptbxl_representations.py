import logging

import hydra
from omegaconf import DictConfig

from msr.data.representation.ptbxl import create_ptbxl_representations_dataset
from msr.utils import print_config_tree

log = logging.getLogger(__name__)
log.setLevel(logging.INFO)


@hydra.main(version_base=None, config_path="../../configs/create_representations", config_name="ptbxl")
def main(cfg: DictConfig):
    log.info("Creating PTB-XL representations dataset")

    print_config_tree(
        cfg, keys=["raw_data", "beats_params", "n_beats", "agg_beat_params", "windows_params", "representation_types"]
    )

    create_ptbxl_representations_dataset(
        representation_types=cfg.representation_types,
        beats_params=cfg.beats_params,
        n_beats=cfg.n_beats,
        agg_beat_params=cfg.agg_beat_params,
        windows_params=cfg.windows_params,
        fs=cfg.raw_data.fs,
    )
    log.info("Finished.")


if __name__ == "__main__":
    main()
