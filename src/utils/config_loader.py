# config_loader.py
import os
import yaml
from functools import lru_cache

workspace_dir = os.getenv('WORKSPACE_DIR')
if workspace_dir:
    os.chdir(workspace_dir)

@lru_cache(maxsize=1)
def load_config(config_path='config.yaml'):
    """
    Loads the YAML configuration file and caches the result for efficient reuse.
    
    Parameters:
        config_path (str): Path to the configuration file.
    
    Returns:
        dict: Configuration parameters loaded from YAML.
    
    Raises:
        FileNotFoundError: If the configuration file does not exist.
    """
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    return config
