import yaml
import os
import logging
from typing import Dict, Any


def load_config(config_path: str) -> Dict[str, Any]:
    logger = logging.getLogger(__name__)

    # Check if file exists
    if not os.path.exists(config_path):
        logger.error(f"Config file not found at {config_path}")
        raise FileNotFoundError(f"Config file not found at {config_path}")

    # Load config file
    try:
        logger.debug(f"Loading config from {config_path}")
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)
        return config
    except yaml.YAMLError as e:
        logger.error(f"Error parsing YAML file: {e}")
        raise
    except Exception as e:
        logger.error(f"Unexpected error loading config: {str(e)}")
        raise