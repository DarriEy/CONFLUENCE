import yaml # type: ignore
from pathlib import Path
from typing import Dict, Any

class ConfigManager:
    def __init__(self, config):
        self.config_file = config
        self.config = self.load_config()

    def load_config(self):
        with open(self.config_file, 'r') as config_file:
            return yaml.safe_load(config_file)

    def get(self, key, default=None):
        return self.config.get(key, default)

    def set(self, key, value):
        self.config[key] = value

    def save(self):
        with open(self.config_file, 'w') as config_file:
            yaml.dump(self.config, config_file)

def get_default_path(config: Dict[str, Any], project_dir: str, path_key: str, default_subpath: str, logger: Any) -> Path:
        """
        Get a path from config or use a default based on the project directory.

        Args:
            config (dict): The configuration dictionary to access
            path_key (str): The key to look up in the config dictionary.
            default_subpath (str): The default subpath to use if the config value is 'default'.
            logger: logger to log any errors in execution

        Returns:
            Path: The resolved path.

        Raises:
            KeyError: If the path_key is not found in the config.
        """
        try:
            path_value = config.get(path_key)
            if path_value == 'default' or path_value is None:
                return project_dir / default_subpath
            return Path(path_value)
        except KeyError:
            logger.error(f"Config key '{path_key}' not found")
            raise