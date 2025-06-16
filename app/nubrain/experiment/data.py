import json
import pickle
from time import time

import h5py

from nubrain.experiment.global_config import GlobalConfig

global_config = GlobalConfig()


def eeg_data_logging(subprocess_params: dict):
    """
    Log experimental data. For now, save to local hdf file. TODO: Send to remote
    backend. To be run in separate process (using multiprocessing).
    """
    # ----------------------------------------------------------------------------------
    # *** Get parameters

    demo_mode = subprocess_params["demo_mode"]
    image_directory = subprocess_params["image_directory"]

    # EEG parameters
    eeg_board_description = subprocess_params["eeg_board_description"]
    eeg_sampling_rate = subprocess_params["eeg_sampling_rate"]
    n_channels_total = subprocess_params["n_channels_total"]
    eeg_channels = subprocess_params["eeg_channels"]
    marker_channel = subprocess_params["marker_channel"]
    eeg_channel_mapping = subprocess_params["eeg_channel_mapping"]
    eeg_device_address = subprocess_params["eeg_device_address"]

    # Timing parameters
    initial_rest_duration = subprocess_params["initial_rest_duration"]
    image_duration = subprocess_params["image_duration"]
    isi_duration = subprocess_params["isi_duration"]
    inter_block_grey_duration = subprocess_params["inter_block_grey_duration"]

    # Experiment structure
    n_blocks = subprocess_params["n_blocks"]
    images_per_block = subprocess_params["images_per_block"]

    # Filter parameters
    # bandstop_low_cutoff_freq = subprocess_params["bandstop_low_cutoff_freq"]
    # bandstop_high_cutoff_freq = subprocess_params["bandstop_high_cutoff_freq"]
    # bandstop_filter_order = subprocess_params["bandstop_filter_order"]
    # bandpass_low_cutoff_freq = subprocess_params["bandpass_low_cutoff_freq"]
    # bandpass_high_cutoff_freq = subprocess_params["bandpass_high_cutoff_freq"]
    # bandpass_filter_order = subprocess_params["bandpass_filter_order"]

    # nubrain_endpoint = subprocess_params["nubrain_endpoint"]
    # nubrain_api_key = subprocess_params["nubrain_api_key"]

    path_out_data = subprocess_params["path_out_data"]

    data_logging_queue = subprocess_params["data_logging_queue"]

    # ----------------------------------------------------------------------------------
    # *** Create and initialize HDF5 file

    experiment_metadata = {
        "config_version": global_config.config_version,
        "demo_mode": demo_mode,
        "image_directory": image_directory,
        "rest_condition_color": global_config.rest_condition_color,
        "stim_start_marker": global_config.stim_start_marker,
        "stim_end_marker": global_config.stim_end_marker,
        "hdf5_dtype": global_config.hdf5_dtype,
        "experiment_start_time": time(),
        # EEG parameters
        "eeg_board_description": eeg_board_description,
        "eeg_sampling_rate": eeg_sampling_rate,
        "n_channels_total": n_channels_total,
        "eeg_channels": eeg_channels,
        "marker_channel": marker_channel,
        "eeg_channel_mapping": eeg_channel_mapping,
        "eeg_device_address": eeg_device_address,
        # Timing parameters
        "initial_rest_duration": initial_rest_duration,
        "image_duration": image_duration,
        "isi_duration": isi_duration,
        "inter_block_grey_duration": inter_block_grey_duration,
        # Experiment structure
        "n_blocks": n_blocks,
        "images_per_block": images_per_block,
        # Filter parameters
        # "bandstop_low_cutoff_freq": bandstop_low_cutoff_freq,
        # "bandstop_high_cutoff_freq": bandstop_high_cutoff_freq,
        # "bandstop_filter_order": bandstop_filter_order,
        # "bandpass_low_cutoff_freq": bandpass_low_cutoff_freq,
        # "bandpass_high_cutoff_freq": bandpass_high_cutoff_freq,
        # "bandpass_filter_order": bandpass_filter_order,
    }

    print(f"Initializing HDF5 file at: {path_out_data}")
    with h5py.File(path_out_data, "w") as file:
        # Write metadata to attributes. Iterate through the dictionary and write each
        # key-value pair.
        for key, value in experiment_metadata.items():
            # If the value is a dictionary itself, we can't write it directly. Serialize
            # it to a JSON string first.
            if isinstance(value, dict):
                # The 'json.dumps' function converts a Python dict to a JSON string.
                file.attrs["experiment_metadata"][key] = json.dumps(value)
            else:
                file.attrs["experiment_metadata"][key] = value

        # Initialize dataset for measurement data. To handle a variable number of
        # timesteps, create a resizable dataset. We specify an initial shape but set the
        # 'maxshape' to allow one of the dimensions to be unlimited (by setting it to
        # None). 'chunks=True' is recommended for resizable datasets for better
        # performance. It lets h5py decide the chunk size.
        file.create_dataset(
            "board_data",
            shape=(n_channels_total, 0),  # Start with 0 timesteps
            maxshape=(n_channels_total, None),  # fixed_channels, unlimited_timesteps
            dtype=global_config.hdf5_dtype,
            chunks=True,
        )

    # ----------------------------------------------------------------------------------
    # ***

    counter = 0

    while True:
        data_to_send = data_logging_queue.get(block=True)

        print(f"Data sender counter: {counter}")
        counter += 1

        if data_to_send is None:
            # Received None. End process.
            print("Ending preprocessing & data saving process.")
            break

        board_data = data_to_send["board_data"]
        metadata = data_to_send["metadata"]
        image_filepath = data_to_send["image_filepath"]

        # ------------------------------------------------------------------------------
        # *** Local data copy

        trial_data = {
            "stimulus_start_time": metadata["stimulus_start_time"],
            "stimulus_end_time": metadata["stimulus_end_time"],
            "stimulus_duration_s": metadata["stimulus_duration_s"],
            "eeg": eeg_data,
            "image_filepath": image_filepath,  # TODO image metadata
        }
        experiment_data["data"].append(trial_data)

    # ----------------------------------------------------------------------------------
    # *** Save local data copy

    print("Save data to disk")

    with open(path_out_data, "wb") as file:
        pickle.dump(experiment_data, file, protocol=pickle.HIGHEST_PROTOCOL)

    # End of data preprocessing process.
