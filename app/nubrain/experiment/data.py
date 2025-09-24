import json
from time import time

import h5py
import numpy as np

from nubrain.experiment.global_config import GlobalConfig
from nubrain.image.tools import load_image_as_bytes, resize_image

global_config = GlobalConfig()


def eeg_data_logging(subprocess_params: dict):
    """
    Log experimental data. Save to local hdf file. To be run in separate process (using
    multiprocessing).
    """
    # ----------------------------------------------------------------------------------
    # *** Get parameters

    image_directory = subprocess_params["image_directory"]

    subject_id = subprocess_params["subject_id"]
    session_id = subprocess_params["session_id"]

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

    utility_frequency = subprocess_params["utility_frequency"]

    # nubrain_endpoint = subprocess_params["nubrain_endpoint"]
    # nubrain_api_key = subprocess_params["nubrain_api_key"]

    path_out_data = subprocess_params["path_out_data"]

    data_logging_queue = subprocess_params["data_logging_queue"]

    # ----------------------------------------------------------------------------------
    # *** Create and initialize HDF5 file

    experiment_metadata = {
        "config_version": global_config.config_version,
        "subject_id": subject_id,
        "session_id": session_id,
        "image_directory": image_directory,
        "rest_condition_color": global_config.rest_condition_color,
        "stim_start_marker": global_config.stim_start_marker,
        "stim_end_marker": global_config.stim_end_marker,
        "hdf5_dtype": global_config.hdf5_dtype,
        "max_img_storage_dimension": global_config.max_img_storage_dimension,
        "experiment_start_time": time(),
        # EEG parameters
        "eeg_board_description": eeg_board_description,
        "eeg_sampling_rate": eeg_sampling_rate,
        "n_channels_total": n_channels_total,
        "eeg_channel_mapping": eeg_channel_mapping,
        # Timing parameters
        "initial_rest_duration": initial_rest_duration,
        "image_duration": image_duration,
        "isi_duration": isi_duration,
        "inter_block_grey_duration": inter_block_grey_duration,
        # Experiment structure
        "n_blocks": n_blocks,
        "images_per_block": images_per_block,
        # Misc
        "utility_frequency": utility_frequency,
    }

    # Parameters not used by DSI-24, for compatibility with Cyton board.
    if eeg_device_address is not None:
        experiment_metadata["eeg_device_address"] = eeg_device_address
    if eeg_channels is not None:
        experiment_metadata["eeg_channels"] = eeg_channels
    if marker_channel is not None:
        experiment_metadata["marker_channel"] = marker_channel

    print(f"Initializing HDF5 file at: {path_out_data}")
    with h5py.File(path_out_data, "w") as file:
        # ------------------------------------------------------------------------------
        # *** Initialize hdf5 dataset for metadata

        # Create group for metadata.
        metadata_group = file.create_group("metadata")

        # Iterate over the Python dictionary and save each item as an attribute of the
        # "metadata" group.
        for key, value in experiment_metadata.items():
            # HDF5 attributes have limitations on data types. Complex types like
            # dictionaries or tuples are not natively supported. We check if the value
            # is a type that needs to be converted to a string. JSON is a convenient
            # format for this serialization.
            if isinstance(value, (dict, list, tuple)):
                # Serialize the complex type into a JSON string.
                metadata_group.attrs[key] = json.dumps(value)
            else:
                metadata_group.attrs[key] = value

        # ------------------------------------------------------------------------------
        # *** Initialize hdf5 dataset for EEG data

        # Initialize dataset for EEG and additional channels. To handle a variable
        # number of timesteps, create a resizable dataset. We specify an initial shape
        # but set the 'maxshape' to allow one of the dimensions to be unlimited (by
        # setting it to None). 'chunks=True' is recommended for resizable datasets for
        # better performance. It lets h5py decide the chunk size.

        file.create_dataset(
            "eeg_data",
            shape=(n_channels_total, 0),
            maxshape=(n_channels_total, None),  # fixed_channels, unlimited_timesteps
            dtype=global_config.hdf5_dtype,
            chunks=True,
        )

        file.create_dataset(
            "eeg_timestamps",
            shape=(0,),
            maxshape=(None,),
            dtype="float64",  # LSL timestamps
            chunks=True,
        )

        file.create_dataset(
            "marker_data",
            shape=(0,),
            maxshape=(None,),
            dtype="float64",
            chunks=True,
        )

        # ------------------------------------------------------------------------------
        # *** Initialize hdf5 dataset for stimulus data

        # Define the compound datatype for stimulus data. This is like defining the
        # columns of a table. Use a special vlen dtype for the variable-sized image
        # data.
        stimulus_dtype = np.dtype(
            [
                ("stimulus_start_time", np.uint64),
                ("stimulus_end_time", np.uint64),
                ("stimulus_duration_s", np.float32),
                ("image_file_path", h5py.string_dtype(encoding="utf-8")),
                ("image_category", h5py.string_dtype(encoding="utf-8")),
                # ("image_description", h5py.string_dtype(encoding="utf-8")),
                (
                    "image_bytes",
                    h5py.vlen_dtype(np.uint8),
                ),  # For variable-length byte arrays
            ]
        )

        n_images = n_blocks * images_per_block

        file.create_dataset(
            "stimulus_data",
            (n_images,),
            dtype=stimulus_dtype,
        )

    # ----------------------------------------------------------------------------------
    # *** Experiment loop

    stimulus_counter = 0

    while True:
        new_data = data_logging_queue.get(block=True)

        if new_data is None:
            # Received None. End process.
            print("Ending preprocessing & data saving process.")
            break

        data_type = new_data["type"]

        with h5py.File(path_out_data, "a") as file:
            # --------------------------------------------------------------------------
            # *** Write EEG data to hdf5 file

            if data_type == "eeg":
                new_eeg_data = new_data.get("eeg_data")
                new_timestamps = new_data.get("eeg_timestamps")

                if new_eeg_data is not None and new_eeg_data.size > 0:
                    # Write EEG data.
                    hdf5_eeg_data = file["eeg_data"]
                    n_existing = hdf5_eeg_data.shape[1]
                    n_new = new_eeg_data.shape[1]
                    hdf5_eeg_data.resize(n_existing + n_new, axis=1)
                    hdf5_eeg_data[:, n_existing:] = new_eeg_data

                    # Write EEG timestamps.
                    hdf5_timestamps = file["eeg_timestamps"]
                    n_existing_ts = hdf5_timestamps.shape[0]
                    hdf5_timestamps.resize(n_existing_ts + n_new, axis=0)
                    hdf5_timestamps[n_existing_ts:] = new_timestamps

            # --------------------------------------------------------------------------
            # *** Write stimulus markers to hdf5 file

            elif data_type == "marker":
                marker_value = new_data.get("marker_value")
                marker_timestamp = new_data.get("timestamp")

                if marker_value is not None:
                    hdf5_marker_data = file["marker_data"]
                    n_existing = hdf5_marker_data.shape[0]
                    hdf5_marker_data.resize(n_existing + 1, axis=0)
                    hdf5_marker_data[n_existing] = (marker_value, marker_timestamp)

            # --------------------------------------------------------------------------
            # *** Write stimulus data to hdf5 file

            elif data_type == "stimulus":
                new_stimulus_data = new_data.get("stimulus_data")

                if new_stimulus_data is not None:
                    hdf5_stimulus_data = file["stimulus_data"]

                    image_file_path = new_stimulus_data["image_file_path"]
                    image_bytes = load_image_as_bytes(image_path=image_file_path)
                    image_bytes = resize_image(image_bytes=image_bytes)

                    stimulus_start_time = new_stimulus_data["stimulus_start_time"]
                    stimulus_end_time = new_stimulus_data["stimulus_end_time"]
                    stimulus_duration_s = new_stimulus_data["stimulus_duration_s"]
                    image_file_path = new_stimulus_data["image_file_path"]
                    image_category = new_stimulus_data["image_category"]
                    # image_description = new_stimulus_data["image_description"]

                    data_to_write = np.empty((1,), dtype=stimulus_dtype)
                    data_to_write[0]["stimulus_start_time"] = stimulus_start_time
                    data_to_write[0]["stimulus_end_time"] = stimulus_end_time
                    data_to_write[0]["stimulus_duration_s"] = stimulus_duration_s
                    data_to_write[0]["image_file_path"] = image_file_path
                    data_to_write[0]["image_category"] = image_category
                    # data_to_write[0]["image_description"] = image_description
                    # The image data is stored as a numpy array of bytes (uint8).
                    data_to_write[0]["image_bytes"] = np.frombuffer(
                        image_bytes,
                        dtype=np.uint8,
                    )

                    # Write the structured array to the dataset.
                    hdf5_stimulus_data[stimulus_counter] = data_to_write

                    print(f"Stimulus counter: {stimulus_counter}")
                    stimulus_counter += 1

    # End of data preprocessing process.
