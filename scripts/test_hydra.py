from pathlib import Path

import hydra
from hydra.core.config_store import ConfigStore

from config import TESTConfig

config_store = ConfigStore.instance()
config_store.store(name="test_config", node=TESTConfig)

@hydra.main(config_path="../conf", config_name="config", version_base=None)
def main(cfg: TESTConfig):
    print(cfg.environment.source)
    return

if __name__ == "__main__":
    main()