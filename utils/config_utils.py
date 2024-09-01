import yaml # type: ignore
from pathlib import Path

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