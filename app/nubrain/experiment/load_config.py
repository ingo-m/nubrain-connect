import os
from dataclasses import dataclass, field, fields
from typing import Dict, get_origin

import yaml


@dataclass(frozen=True)
class EegExperimentConfig:
    """
    Validate configuration for EEG experiment.

    We do not use pydantic, because of the breaking changes introduced in pydantic 2 and
    the problems with dependency resolution for libraries that depend either on pydantic
    1 or 2. Pydantic is the path to dependency hell.

    Setting `frozen=True` makes instances immutable, to prevent accidental changes at
    runtime.
    """

    subject_id: str
    session_id: str

    output_directory: str
    image_directory: str

    utility_frequency: float

    # Timing parameters
    initial_rest_duration: float
    image_duration: float
    isi_duration: float
    isi_jitter: float
    inter_block_grey_duration: float

    # Experiment structure
    n_blocks: int
    images_per_block: int

    device_type: str
    lsl_stream_name: str

    eeg_device_address: str = ""  # Make optional with default

    # Use default_factory for mutable types
    eeg_channel_mapping: Dict[int, str] = field(default_factory=dict)

    def __post_init__(self):
        """
        Validation after the object has been initialized.
        """

        # Runtime type validation. Iterate over all fields defined in the dataclass.
        for f in fields(self):
            value = getattr(self, f.name)

            # Use `get_origin` to handle generic types like `Dict[int, str]`.
            # `get_origin(f.type)` will be `dict`, while for `int` it will be `None`.
            # The `or` operator provides a fallback to the type itself for non-generics.
            check_type = get_origin(f.type) or f.type

            if not isinstance(value, check_type):
                raise TypeError(
                    f"Invalid type for '{f.name}'. "
                    f"Expected {check_type.__name__}, but got {type(value).__name__}."
                )

        # The loop above checked that `eeg_channel_mapping` is a dict. Now we check the
        # contents of the dict.
        if not all(isinstance(k, int) for k in self.eeg_channel_mapping.keys()):
            raise TypeError("All keys in 'eeg_channel_mapping' must be integers.")

        if not all(isinstance(v, str) for v in self.eeg_channel_mapping.values()):
            raise TypeError("All values in 'eeg_channel_mapping' must be strings.")

        # Validate device_type.
        valid_devices = ["cyton", "dsi24", "synthetic"]
        if self.device_type not in valid_devices:
            raise ValueError(
                f"device_type must be one of {valid_devices}, got {self.device_type}"
            )

        if self.device_type == "synthetic":
            print("WARNING: USING SYNTHETIC DEVICE (DEMO MODE)")

        print("Configuration successfully loaded and validated.")


def load_config_yaml(*, yaml_file_path: str):
    """
    Load yaml file with settings for nubrain EEG experiment.
    """
    if not os.path.isfile(yaml_file_path):
        raise AssertionError(f"Config file not found: {yaml_file_path}")

    with open(yaml_file_path, "r") as file:
        config_dict = yaml.safe_load(file)

    # Validate config.
    config_dataclass = EegExperimentConfig(**config_dict)

    return config_dict
