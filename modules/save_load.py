# Load settings from a JSON file
import json


def load_settings(filepath):
    try:
        with open(filepath, 'r') as f:
            settings = json.load(f)
        return settings
    except FileNotFoundError:
        return {}


def save_settings(settings, filepath):
    with open(filepath, 'w') as f:
        json.dump(settings, f, indent=4)