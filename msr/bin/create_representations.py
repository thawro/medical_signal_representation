import logging
import warnings

import hydra
from omegaconf import DictConfig, OmegaConf

from msr.utils import CONFIG_PATH, print_config_tree

warnings.filterwarnings("ignore", module="numpy")  # TODO: Not a good idea to filter out all warnings


log = logging.getLogger(__name__)
log.setLevel(logging.INFO)


@hydra.main(version_base=None, config_path=CONFIG_PATH / "create_representations", config_name="create_representations")
def main(cfg: DictConfig):
    log.info("Creating representations dataset")
    print_config_tree(cfg, keys="all")

    dataset_provider = hydra.utils.instantiate(cfg.dataset_provider)
    dataset_provider.create_dataset(n_jobs=cfg.n_jobs, batch_size=cfg.batch_size, splits=cfg.splits)
    dataset_provider.concat_data_files()

    log.info("Representasions dataset creation finished")
    OmegaConf.save(cfg, dataset_provider.representations_path / "config.yaml")
    log.info("Config file saved in representations directory")
    log.info("Finished")


if __name__ == "__main__":
    main()
