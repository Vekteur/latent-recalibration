import sys

from omegaconf import OmegaConf

from moc.configs.config import get_config
from moc.parallel_runner import run_all
from moc.utils import configure_logging, print_config


def main():
    configure_logging()
    # Create config with default values overwritten by CLI arguments
    config = OmegaConf.from_cli(sys.argv)
    config = get_config(config)
    # Pretty print config using Rich library
    if config.get('print_config'):
        print_config(config, resolve=True)
    # Run experiments
    return run_all(config)


if __name__ == '__main__':
    main()
