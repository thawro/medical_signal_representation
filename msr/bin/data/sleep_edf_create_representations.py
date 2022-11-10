import logging

import hydra
from omegaconf import DictConfig

from msr.data.representation.sleep_edf import create_sleep_edf_representations_dataset
from msr.utils import print_config_tree

log = logging.getLogger(__name__)
log.setLevel(logging.INFO)


@hydra.main(version_base=None, config_path="../../configs/data", config_name="sleep_edf")
def main(cfg: DictConfig):
    log.info("Creating Sleep EDF representations dataset")

    print_config_tree(cfg, keys=["raw_data", "create_representations"])

    create_sleep_edf_representations_dataset(
        representation_types=cfg.create_representations.representation_types,
        windows_params=cfg.create_representations.windows_params,
        fs=cfg.raw_data.fs,
    )
    log.info("Finished.")


if __name__ == "__main__":
    main()
