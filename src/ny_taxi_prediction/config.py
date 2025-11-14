import yaml
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent.parent
CONFIG_PATH = BASE_DIR / "params.yaml"

def load_config():
    with open(CONFIG_PATH, "r") as f:
        config = yaml.safe_load(f)
    return config

if __name__ == "__main__":
    cfg = load_config()
    print(cfg['model']['learning_rate'])