import yaml
from copy import deepcopy

def deep_update(base_dict, update_dict):
    """Recursively update nested dictionaries."""
    for key, value in update_dict.items():
        if isinstance(value, dict) and key in base_dict and isinstance(base_dict[key], dict):
            deep_update(base_dict[key], value)
        else:
            base_dict[key] = value
    return base_dict

def read_yaml(file_path):
    # Load base configuration
    with open("config/base.yaml", "r") as f:
         base_config = yaml.safe_load(f)
    # Load override configuration
    with open(file_path) as f:
     override_config = yaml.safe_load(f)
    # Make a deep copy to avoid modifying the original
    config = deepcopy(base_config)
    # Recursively update with override config
    deep_update(config, override_config)
    return config

def override_yaml(file_path, override):
    # Load base configuration
    with open("config/base.yaml", "r") as f:
        base_config = yaml.safe_load(f)
    # Load override configuration
    with open(file_path, "r") as f:
        override_config = yaml.safe_load(f)
    # Make a deep copy to avoid modifying the original
    config = deepcopy(base_config)
    # First update with file overrides
    deep_update(config, override_config)
    # Then update with parameter overrides
    deep_update(config, override)
    return config