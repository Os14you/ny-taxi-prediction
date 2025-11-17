import yaml

def load_config(path: str = "params.yaml") -> dict:
    """Load project parameters from the params.yaml file."""
    with open(path, "r") as f:
        params = yaml.safe_load(f)
    return params

if __name__ == "__main__":
    cfg = load_config()
    print(cfg['preprocessing']['random_state'])