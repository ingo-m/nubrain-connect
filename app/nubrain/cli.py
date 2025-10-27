import argparse

from nubrain.experiment.load_config import load_config_yaml

# Wrap these imports in try, so that the other modules can be imported without
# dependency on pylsl for demo mode.
try:
    from nubrain.experiment.main import experiment
    from nubrain.experiment_eeg_to_image_v1.load_config import (
        load_config_yaml_eeg_to_image_v1,
    )
    from nubrain.experiment_eeg_to_image_v1.main import experiment_eeg_to_image_v1
    from nubrain.experiment_eeg_to_image_v1.main_autoregressive import (
        experiment_eeg_to_image_v1_autoregressive,
    )
except Exception as e:
    experiment = None
    load_config_yaml_eeg_to_image_v1 = None
    experiment_eeg_to_image_v1 = None
    experiment_eeg_to_image_v1_autoregressive = None
    print(f"Failed to import nubrain main module: {e}")

from nubrain.live_demo.main import run_live_demo


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

    parser.add_argument(
        "--live_demo",
        action="store_true",
        help="Demo mode.",
    )

    args = parser.parse_args()

    print("nubrain")
    print(f"Configuration file provided: {args.config}")

    input_file_path = args.config

    # Whether to run live EEG to image generation. Set to True if the flag is present.
    eeg_to_image_v1 = args.eeg_to_image_v1

    autoregressive = args.autoregressive

    live_demo = args.live_demo

    # Load EEG experiment config from yaml file.
    if eeg_to_image_v1:
        # Live EEG to image generation mode. Use corresponding config file loading
        # function (different parameters than regular data collection).
        config = load_config_yaml_eeg_to_image_v1(yaml_file_path=input_file_path)
    elif live_demo:
        pass
    else:
        # Regular data collection mode.
        config = load_config_yaml(yaml_file_path=input_file_path)

    if eeg_to_image_v1:
        if autoregressive:
            # Autoregressive live EEG to image generation mode (use previous
            # reconstructed image as next stimulus).
            experiment_eeg_to_image_v1_autoregressive(config=config)
        else:
            # Live EEG to image generation mode.
            experiment_eeg_to_image_v1(config=config)
    elif live_demo:
        run_live_demo(cache=input_file_path)  # Pickle file path
    else:
        # Regular data collection mode.
        experiment(config=config)


if __name__ == "__main__":
    main()
