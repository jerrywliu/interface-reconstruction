import yaml

def read_yaml(file_path):
    # Load base configuration
    with open("config/base.yaml", "r") as f:
         base_config = yaml.safe_load(f)
    # Load override configuration
    with open(file_path) as f:
        override_config = yaml.safe_load(f)
    config = {**base_config, **override_config}
    return config

def override_yaml(file_path, override):
    # Load base configuration
    with open("config/base.yaml", "r") as f:
         base_config = yaml.safe_load(f)
    # Load override configuration
    with open(file_path, "r") as f:
         override_config = yaml.safe_load(f)
    config = {**base_config, **override_config, **override}
    return config
