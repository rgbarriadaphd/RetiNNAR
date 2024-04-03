"""
# Author = ruben
# Date: 2/4/24
# Project: RetiNNAR
# File: utils.py

Description: Utils file for common functionalities.
"""
import json
import os.path


class Utils:
    def __init__(self, config_path: str):
        self._retrieve_root_path()
        self._retrieve_config_from_json(config_path)

    def _retrieve_root_path(self):
        """Retrieve absolute project root path"""
        self._root_path = os.path.normpath(os.path.join(os.path.dirname(__file__), ".."))

    def _retrieve_config_from_json(self, config_path: str):
        """Get the config from a json file"""
        json_file = os.path.join(self._root_path, config_path)
        with open(json_file, 'r') as config_file:
            try:
                self._config = json.load(config_file)
            except ValueError:
                print(f'INVALID JSON file format: {json_file}')
                exit(-1)

    def build_path(self, partial: str) -> str:
        """Builds and verify path from project root path"""
        path = os.path.join(self._root_path, partial)
        assert os.path.exists(path), f'File: {path} does not exists'
        return path

    def get_root_path(self) -> str:
        """Return project root path"""
        return self._root_path

    def get_config(self) -> dict:
        """Return config file"""
        return self._config


if __name__ == '__main__':
    utils = Utils("config/retinnar.json")
    print(utils.get_root_path())
    print(utils.get_config())
