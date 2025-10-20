import argparse

from nubrain.experiment.load_config import load_config_yaml
from nubrain.experiment.main import experiment
from nubrain.experiment_eeg_to_image_v1.load_config import (
    load_config_yaml_eeg_to_image_v1,
)
from nubrain.experiment_eeg_to_image_v1.main import experiment_eeg_to_image_v1
from nubrain.experiment_eeg_to_image_v1.main_autoregressive import (
    experiment_eeg_to_image_v1_autoregressive,
)


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

    parser.add_argument(
        "--autoregressive",
        action="store_true",
        help="Use previous reconstructed image as the next stimulus.",
    )

    args = parser.parse_args()

    print("nubrain")
    print(f"Configuration file provided: {args.config}")

    yaml_file_path = args.config

    # Whether to run live EEG to image generation. Set to True if the flag is present.
    eeg_to_image_v1 = args.eeg_to_image_v1

    autoregressive = args.autoregressive

    # Load EEG experiment config from yaml file.
    if eeg_to_image_v1:
        # Live EEG to image generation mode. Use corresponding config file loading
        # function (different parameters than regular data collection).
        config = load_config_yaml_eeg_to_image_v1(yaml_file_path=yaml_file_path)
    else:
        # Regular data collection mode.
        config = load_config_yaml(yaml_file_path=yaml_file_path)

    if eeg_to_image_v1:
        if autoregressive:
            # Autoregressive live EEG to image generation mode (use previous
            # reconstructed image as next stimulus).
            experiment_eeg_to_image_v1_autoregressive(config=config)
        else:
            # Live EEG to image generation mode.
            experiment_eeg_to_image_v1(config=config)
    else:
        # Regular data collection mode.
        experiment(config=config)


if __name__ == "__main__":
    main()
