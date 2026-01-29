import argparse

from nubrain.experiment_image.load_config import load_config_image_yaml
from nubrain.experiment_text.load_config import load_config_text_yaml

# Wrap these imports in try, so that the other modules can be imported without
# dependency on pylsl for demo mode.
try:
    from nubrain.experiment_eeg_to_image_v1.load_config import (
        load_config_yaml_eeg_to_image_v1,
    )
    from nubrain.experiment_eeg_to_image_v1.main import experiment_eeg_to_image_v1
    from nubrain.experiment_eeg_to_image_v1.main_autoregressive import (
        experiment_eeg_to_image_v1_autoregressive,
    )
    from nubrain.experiment_image.main import experiment_image
    from nubrain.experiment_text.main import experiment_text
except Exception as e:
    experiment_image = None
    experiment_text = None
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

    # Which experimental mode to use. Options:
    # - "data_collection_image": Data collection mode for image stimuli.
    # - "data_collection_text":  Data collection mode for text stimuli.
    # - "eeg_to_image":  After presenting each image, directly reconstruct the image,
    #   then show the next image.
    # - "eeg_to_image_autoregressive": After presenting an image, directly reconstruct
    #   the image, and then show the reconstructed image as the next stimulus.
    # - "eeg_to_image_live_demo": Use cache.
    parser.add_argument(
        "--mode",
        type=str,
        default="",
        help="Which experimental mode to use",
    )

    args = parser.parse_args()

    print("nubrain")
    print(f"Configuration file provided: {args.config}")

    input_file_path = args.config

    mode = args.mode

    # Load EEG experiment config from yaml file.
    if mode == "data_collection_image":
        # Data collection mode, image stimuli.
        config = load_config_image_yaml(yaml_file_path=input_file_path)
    elif mode == "data_collection_text":
        # Data collection mode, text stimuli.
        config = load_config_text_yaml(yaml_file_path=input_file_path)
    elif mode in ["eeg_to_image", "eeg_to_image_autoregressive"]:
        # Live EEG to image generation mode. Use corresponding config file loading
        # function (different parameters than regular data collection).
        config = load_config_yaml_eeg_to_image_v1(yaml_file_path=input_file_path)
    elif mode == "eeg_to_image_live_demo":
        pass
    else:
        raise AssertionError(f"Unknown experimental mode: {mode}")

    # Run experiment.
    if mode == "data_collection_image":
        experiment_image(config=config)
    elif mode == "data_collection_text":
        experiment_text(config=config)
    elif mode == "eeg_to_image":
        experiment_eeg_to_image_v1(config=config)
    elif mode == "eeg_to_image_autoregressive":
        experiment_eeg_to_image_v1_autoregressive(config=config)
    elif mode == "eeg_to_image_live_demo":
        run_live_demo(cache=input_file_path)  # Pickle file path


if __name__ == "__main__":
    main()
