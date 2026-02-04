import json
from time import time

import h5py
import numpy as np

from nubrain.experiment_text.global_config import GlobalConfig

global_config = GlobalConfig()


def eeg_data_logging(subprocess_params: dict):
    """
    Log experimental data. Save to local hdf file. To be run in separate process (using
    multiprocessing).
    """
    # ----------------------------------------------------------------------------------
    # *** Get parameters

    path_text = subprocess_params["path_text"]

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
    stimulus_duration = subprocess_params["stimulus_duration"]
    isi_duration = subprocess_params["isi_duration"]
    isi_jitter = subprocess_params["isi_jitter"]
    isi_extension_target = subprocess_params["isi_extension_target"]
    inter_block_rest_duration = subprocess_params["inter_block_rest_duration"]

    # Experiment structure
    word_idx_start = subprocess_params["word_idx_start"]
    n_words_to_show = subprocess_params["n_words_to_show"]
    n_target_events = subprocess_params["n_target_events"]
    min_distance_targets = subprocess_params["min_distance_targets"]
    stimuli_per_block = subprocess_params["stimuli_per_block"]
    stimulus_font_size = subprocess_params["stimulus_font_size"]

    # Text and targets
    text = subprocess_params["text"]  # List of str
    is_target = subprocess_params["is_target"]  # List of bool

    # Misc
    utility_frequency = subprocess_params["utility_frequency"]
    path_out_data = subprocess_params["path_out_data"]
    data_logging_queue = subprocess_params["data_logging_queue"]

    # ----------------------------------------------------------------------------------
    # *** Create and initialize HDF5 file

    experiment_metadata = {
        "config_version": global_config.config_version,
        "subject_id": subject_id,
        "session_id": session_id,
        "path_text": path_text,
        "rest_condition_color": global_config.rest_condition_color,
        "stim_start_marker": global_config.stim_start_marker,
        "stim_end_marker": global_config.stim_end_marker,
        "hdf5_dtype": global_config.hdf5_dtype,
        "experiment_start_time": time(),
        # EEG parameters
        "eeg_board_description": eeg_board_description,
        "eeg_sampling_rate": eeg_sampling_rate,
        "n_channels_total": n_channels_total,
        "eeg_channel_mapping": eeg_channel_mapping,
        # Timing parameters
        "initial_rest_duration": initial_rest_duration,
        "stimulus_duration": stimulus_duration,
        "isi_duration": isi_duration,
        "isi_jitter": isi_jitter,
        "isi_extension_target": isi_extension_target,
        "inter_block_rest_duration": inter_block_rest_duration,
        # Experiment structure
        "word_idx_start": word_idx_start,
        "n_words_to_show": n_words_to_show,
        "n_target_events": n_target_events,
        "min_distance_targets": min_distance_targets,
        "stimuli_per_block": stimuli_per_block,
        "stimulus_font_size": stimulus_font_size,
        # Text and targets
        "text": text,  # List of str
        "is_target": is_target,  # List of bool
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
            shape=(2, 0),  # timestamp, marker value
            maxshape=(2, None),
            dtype="float64",
            chunks=True,
        )

        # ------------------------------------------------------------------------------
        # *** Initialize hdf5 dataset for stimulus data

        # Define the compound datatype for stimulus data. This is like defining the
        # columns of a table.
        stimulus_dtype = np.dtype(
            [
                ("stimulus_start_time", np.float64),
                ("stimulus_end_time", np.float64),
                ("stimulus_duration_s", np.float64),
                ("word", h5py.string_dtype(encoding="utf-8")),
                ("is_target_event", np.bool),
                ("response_time_s", np.float64),
            ]
        )

        file.create_dataset(
            "stimulus_data",
            (len(text),),
            dtype=stimulus_dtype,
        )

        # ------------------------------------------------------------------------------
        # *** Initialize hdf5 dataset for behavioural data

        behavioural_dtype = np.dtype(
            [
                ("n_total_targets", np.int64),
                ("n_hits", np.int64),
                ("n_misses", np.int64),
                ("n_false_alarms", np.int64),
            ]
        )

        file.create_dataset(
            "behavioural_data",
            (1,),
            dtype=behavioural_dtype,
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
                    n_existing = hdf5_marker_data.shape[1]
                    hdf5_marker_data.resize(n_existing + 1, axis=1)
                    hdf5_marker_data[:, n_existing] = (marker_timestamp, marker_value)

            # --------------------------------------------------------------------------
            # *** Write stimulus data to hdf5 file

            elif data_type == "stimulus":
                new_stimulus_data = new_data.get("stimulus_data")

                if new_stimulus_data is not None:
                    hdf5_stimulus_data = file["stimulus_data"]

                    stimulus_start_time = new_stimulus_data["stimulus_start_time"]
                    stimulus_end_time = new_stimulus_data["stimulus_end_time"]
                    stimulus_duration_s = new_stimulus_data["stimulus_duration_s"]
                    word = new_stimulus_data["word"]
                    is_target_event = new_stimulus_data["is_target_event"]
                    response_time_s = new_stimulus_data["response_time_s"]

                    data_to_write = np.empty((1,), dtype=stimulus_dtype)
                    data_to_write[0]["stimulus_start_time"] = stimulus_start_time
                    data_to_write[0]["stimulus_end_time"] = stimulus_end_time
                    data_to_write[0]["stimulus_duration_s"] = stimulus_duration_s
                    data_to_write[0]["word"] = word
                    data_to_write[0]["is_target_event"] = is_target_event
                    data_to_write[0]["response_time_s"] = response_time_s

                    # Write the structured array to the dataset.
                    hdf5_stimulus_data[stimulus_counter] = data_to_write

                    print(f"Stimulus counter: {stimulus_counter}")
                    stimulus_counter += 1

            # --------------------------------------------------------------------------
            # *** Write behavioural data to hdf5 file

            elif data_type == "behavioural":
                new_behavioural_data = new_data.get("behavioural_data")

                if new_behavioural_data is not None:
                    hdf5_behavioural_data = file["behavioural_data"]

                    n_total_targets = new_behavioural_data["n_total_targets"]
                    n_hits = new_behavioural_data["n_hits"]
                    n_misses = new_behavioural_data["n_misses"]
                    n_false_alarms = new_behavioural_data["n_false_alarms"]

                    data_to_write = np.empty((1,), dtype=behavioural_dtype)

                    data_to_write[0]["n_total_targets"] = n_total_targets
                    data_to_write[0]["n_hits"] = n_hits
                    data_to_write[0]["n_misses"] = n_misses
                    data_to_write[0]["n_false_alarms"] = n_false_alarms

                    # Write the structured array to the dataset.
                    hdf5_behavioural_data[0] = data_to_write


# End of data preprocessing process.
