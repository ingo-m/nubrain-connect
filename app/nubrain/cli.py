import argparse

from nubrain.experiment.load_config import load_config_yaml
from nubrain.experiment.main import experiment


def main():
    """
    Main entry point for the nubrain command-line application.
    """
    # Initialize the parser.
    parser = argparse.ArgumentParser(description="nubrain command-line interface.")

    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to the configuration YAML file.",
    )

    args = parser.parse_args()

    print("nubrain")
    print(f"Configuration file provided: {args.config}")

    # Load EEG experiment config from yaml file.
    yaml_file_path = args.config
    config = load_config_yaml(yaml_file_path=yaml_file_path)
    experiment(config=config)


if __name__ == "__main__":
    main()
