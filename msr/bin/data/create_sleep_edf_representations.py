import logging

import hydra
from omegaconf import DictConfig

from msr.data.representation.sleep_edf import create_sleep_edf_representations_dataset
from msr.utils import print_config_tree

log = logging.getLogger(__name__)
log.setLevel(logging.INFO)


@hydra.main(version_base=None, config_path="../../configs/create_representations", config_name="sleep_edf")
def main(cfg: DictConfig):
    log.info("Creating Sleep EDF representations dataset")

    print_config_tree(cfg, keys=["raw_data", "representation_types", "windows_params"])

    create_sleep_edf_representations_dataset(
        representation_types=cfg.representation_types,
        windows_params=cfg.windows_params,
        fs=cfg.raw_data.fs,
    )
    log.info("Finished.")


if __name__ == "__main__":
    main()
