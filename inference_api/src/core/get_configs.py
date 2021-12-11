import os
import json


class InferenceConfig():

    def load_configuration(self, config_name: str) -> dict:
        path = f'core/configs/{config_name}'
        if os.path.exists(path):
            with open(path) as file:
                return json.load(file)
        return dict()
