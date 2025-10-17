import argparse

from nubrain.experiment.load_config import load_config_yaml
from nubrain.experiment.main import experiment
from nubrain.experiment_eeg_to_image_v1.load_config import (
    load_config_yaml_eeg_to_image_v1,
)

# from nubrain.experiment_eeg_to_image_v1.main import experiment_eeg_to_image_v1
from nubrain.experiment_eeg_to_image_v1.main_websocket import experiment_eeg_to_image_v1


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

    parser.add_argument(
        "--eeg_to_image_v1",
        action="store_true",
        help="Live EEG to image generation.",
    )

    args = parser.parse_args()

    print("nubrain")
    print(f"Configuration file provided: {args.config}")

    yaml_file_path = args.config

    # Whether to run live EEG to image generation. Set to True if the flag is present.
    eeg_to_image_v1 = args.eeg_to_image_v1

    # Load EEG experiment config from yaml file.
    if eeg_to_image_v1:
        # Live EEG to image generation mode. Use corresponding config file loading
        # function (different parameters than regular data collection).
        config = load_config_yaml_eeg_to_image_v1(yaml_file_path=yaml_file_path)
    else:
        # Regular data collection mode.
        config = load_config_yaml(yaml_file_path=yaml_file_path)

    if eeg_to_image_v1:
        # Live EEG to image generation mode.
        experiment_eeg_to_image_v1(config=config)
    else:
        # Regular data collection mode.
        experiment(config=config)


if __name__ == "__main__":
    main()
