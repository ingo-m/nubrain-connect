"""
This version: Receive images iteratively (looping over diffusion steps) and via
websocket.
"""

import asyncio
import base64
import io
import json
import multiprocessing as mp
import os
import queue
import random
import threading
import traceback
from time import sleep, time

import numpy as np
import pygame
import websockets

from nubrain.device.device_interface import create_eeg_device
from nubrain.experiment.data import eeg_data_logging
from nubrain.experiment.global_config import GlobalConfig
from nubrain.experiment.randomize_conditions import (
    create_balanced_list,
    sample_next_image,
    shuffle_with_repetitions,
)
from nubrain.experiment_eeg_to_image_v1.tone import generate_tone
from nubrain.image.tools import (
    get_all_images,
    load_and_scale_image,
    scale_image_surface,
)
from nubrain.misc.datetime import get_formatted_current_datetime

mp.set_start_method("spawn", force=True)  # Necessary on if running on windows?


def websocket_client_thread(uri, request_json, image_queue):
    """
    Handle WebSocket communication in a separate thread.

    Connects to the WebSocket, sends data, and puts received images into a queue.
    """

    async def client_logic():
        try:
            async with websockets.connect(uri) as websocket:
                # Send the EEG data
                await websocket.send(request_json)

                # Listen for incoming image messages
                while True:
                    try:
                        message_str = await websocket.recv()
                        message = json.loads(message_str)

                        if "error" in message:
                            print(f"Server error: {message['error']}")
                            image_queue.put(None)  # Signal error
                            break

                        # Put the received data into the thread-safe queue
                        image_queue.put(message)

                        if message.get("step") == "final":
                            break  # End of stream

                    except websockets.exceptions.ConnectionClosed:
                        print("Connection closed by server.")
                        break
        except Exception as e:
            print(f"WebSocket client error: {e}")
            image_queue.put(None)  # Signal error

    # Run the async logic in a new event loop for this thread
    asyncio.run(client_logic())


def experiment_eeg_to_image_v1(config: dict):
    # ----------------------------------------------------------------------------------
    # *** Get config

    device_type = config["device_type"]
    lsl_stream_name = config.get("lsl_stream_name", "DSI-24")

    subject_id = config["subject_id"]
    session_id = config["session_id"]

    output_directory = config["output_directory"]
    image_directory = config["image_directory"]

    eeg_channel_mapping = config.get("eeg_channel_mapping", None)

    utility_frequency = config["utility_frequency"]

    initial_rest_duration = config["initial_rest_duration"]
    pre_stimulus_interval = config["pre_stimulus_interval"]
    image_duration = config["image_duration"]
    post_stimulus_interval = config["post_stimulus_interval"]
    generated_image_duration = config["generated_image_duration"]
    isi_jitter = config["isi_jitter"]
    inter_block_grey_duration = config["inter_block_grey_duration"]

    n_blocks = config["n_blocks"]
    images_per_block = config["images_per_block"]

    api_endpoint = config["api_endpoint"]

    output_dir_images = config["output_dir_images"]

    eeg_device_address = config.get("eeg_device_address", None)

    global_config = GlobalConfig()

    # ----------------------------------------------------------------------------------
    # *** Test if output path exists

    if not os.path.isdir(output_directory):
        raise AssertionError(f"Target directory does not exist: {output_directory}")

    current_datetime = get_formatted_current_datetime()
    path_out_data = os.path.join(output_directory, f"eeg_session_{current_datetime}.h5")

    if os.path.isfile(path_out_data):
        raise AssertionError(f"Target file already exists: {path_out_data}")

    # ----------------------------------------------------------------------------------
    # *** Get input images & their categories

    images_and_categories = get_all_images(image_directory=image_directory)

    if not images_and_categories:
        raise AssertionError(f"Found no images at {image_directory}")
    print(f"Found {len(images_and_categories)} images")

    # ----------------------------------------------------------------------------------
    # *** Create pseudo-random condition order

    # List with all unique image categories (e.g. `["apple", "banana", ...]`).
    image_categories = list(set([x["image_category"] for x in images_and_categories]))
    random.shuffle(image_categories)

    # Order of image categories.
    trial_order = create_balanced_list(
        image_categories=image_categories,
        target_length=n_blocks,
    )
    random.shuffle(trial_order)

    trial_order = shuffle_with_repetitions(
        list_with_duplicates=trial_order,
        repetitions=0,
        minimize_runs=True,
    )

    # Mapping from image categories to image file paths, e.g. `{"apple":
    # ["/path/to/apple_1.png", "/path/to/apple_2.png", ...], "banana":
    # ["/path/to/banana_2.png", ...]}`.
    category_to_filepath = {}
    for item in images_and_categories:
        image_category = item["image_category"]
        image_filepath = item["image_file_path"]
        if image_category in category_to_filepath:
            category_to_filepath[image_category].append(image_filepath)
        else:
            category_to_filepath[image_category] = [image_filepath]

    previous_image_file_path = None

    # ----------------------------------------------------------------------------------
    # *** Prepare EEG measurement

    print(f"Initializing EEG device: {device_type}")

    device_kwargs = {"eeg_channel_mapping": eeg_channel_mapping}
    if device_type in ["cyton", "synthetic"]:
        device_kwargs["eeg_device_address"] = eeg_device_address
    elif device_type == "dsi24":
        device_kwargs["lsl_stream_name"] = lsl_stream_name
    else:
        raise ValueError(f"Unexpected `device_type`: {device_type}")

    eeg_device = create_eeg_device(device_type, **device_kwargs)

    eeg_device.prepare_session()

    # This is a bit clunky. At this point, `eeg_channel_mapping` is None or a dict with
    # a channel mapping from the config yaml file. Overwrite it with the channel mapping
    # from the device (in case of the DSI-24 device, the channel mapping from the device
    # is used in any case).
    eeg_channel_mapping = eeg_device.eeg_channel_mapping

    # Need to start the stream before calling `eeg_device.get_device_info()`, because
    # we retrieve data from board to determine data shape (number of channels).
    eeg_device.start_stream()
    sleep(0.1)

    # Get device info.
    device_info = eeg_device.get_device_info()
    eeg_board_description = device_info["board_description"]
    eeg_sampling_rate = device_info["sampling_rate"]
    eeg_channels = device_info["eeg_channels"]
    marker_channel = device_info["marker_channel"]
    n_channels_total = device_info["n_channels_total"]

    if device_type in ["cyton", "synthetic"]:
        # For Cyton device, we need to get the number of EEG channels from the device
        # (not sure, this might only work after starting the stream).
        eeg_device.eeg_channels = eeg_channels
        eeg_device.timestamp_channel = eeg_board_description["timestamp_channel"]

    print(f"Board: {eeg_board_description['name']}")
    print(f"Sampling Rate: {eeg_sampling_rate} Hz")
    print(f"EEG Channels: {eeg_channels}")
    print(f"Marker Channel: {marker_channel}")
    print(f"EEG Channel Mapping: {eeg_channel_mapping}")

    board_data, board_timestamps = eeg_device.get_board_data()

    print(f"Board data dtype: {board_data.dtype}")
    print(f"Board data shape: {board_data.shape}")
    print(f"Board timestamps shape: {board_timestamps.shape}")

    # ----------------------------------------------------------------------------------
    # *** Start data logging subprocess

    data_logging_queue = mp.Queue()

    subprocess_params = {
        "device_type": device_type,
        "subject_id": subject_id,
        "session_id": session_id,
        "image_directory": image_directory,
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
        "isi_duration": 0.0,  # Dummy value. TODO: Adjust logging for live eeg-to-image
        "inter_block_grey_duration": inter_block_grey_duration,
        # Experiment structure
        "n_blocks": n_blocks,
        "images_per_block": images_per_block,
        # Misc
        "utility_frequency": utility_frequency,
        "path_out_data": path_out_data,
        "data_logging_queue": data_logging_queue,
    }

    logging_process = mp.Process(target=eeg_data_logging, args=(subprocess_params,))
    logging_process.daemon = True
    logging_process.start()

    # ----------------------------------------------------------------------------------
    # *** Start experiment

    running = True
    while running:
        pygame.init()

        # ------------------------------------------------------------------------------
        # *** Prepare audio cue

        pygame.mixer.init(frequency=44100, size=-16, channels=2, buffer=512)

        frequency = 1000  # Pitch of the tone in Hz (e.g., 1000 Hz is a common beep)
        duration = 0.3  # Duration in seconds (half a second)
        amplitude = 0.5  # Volume, from 0.0 to 1.0

        # Play the tone x seconds before the end of the pre-stimulus period.
        tone_pre_stimulus_onset = 0.6

        # Get the sample rate from the mixer settings.
        sample_rate = pygame.mixer.get_init()[0]

        # Generate the tone data.
        tone_data = generate_tone(
            frequency=frequency,
            duration=duration,
            amplitude=amplitude,
            sample_rate=sample_rate,
        )

        # Create a Sound object from the numpy array.
        pure_tone = pygame.sndarray.make_sound(tone_data)

        # ------------------------------------------------------------------------------
        # *** Prepare visual stimulus generation

        # Get screen dimensions and set up full screen.
        screen_info = pygame.display.Info()
        screen_width = screen_info.current_w
        screen_height = screen_info.current_h
        screen = pygame.display.set_mode(
            (screen_width, screen_height), pygame.FULLSCREEN
        )
        pygame.display.set_caption("Image Presentation Experiment")
        pygame.mouse.set_visible(False)

        try:
            # Initial grey screen.
            pygame.time.wait(100)
            screen.fill(global_config.rest_condition_color)
            pygame.display.flip()
            pygame.time.wait(100)
            screen.fill(global_config.rest_condition_color)
            pygame.display.flip()

            # Pause for specified number of milliseconds.
            pygame.time.delay(
                int(round((initial_rest_duration - tone_pre_stimulus_onset) * 1000.0))
            )

            # Clear board buffer.
            _, _ = eeg_device.get_board_data()

            # Block loop.
            for idx_block in range(n_blocks):
                # Average embedding vectors within blocks (across trials). Show x
                # repetitions of the same image, perform inference, and average the
                # embedding vector.

                # Sample the next image.
                next_image_category = trial_order[idx_block]
                next_image_file_path = sample_next_image(
                    next_image_category=next_image_category,
                    category_to_filepath=category_to_filepath,
                    previous_image_file_path=previous_image_file_path,
                )

                # Load the next image.
                image_and_metadata = None
                while image_and_metadata is None:
                    image_and_metadata = load_and_scale_image(
                        image_file_path=next_image_file_path,
                        screen_width=screen_width,
                        screen_height=screen_height,
                    )

                current_image = image_and_metadata["image"]

                # Play tone to cue block start.
                pure_tone.play()
                pygame.time.delay(int(round(tone_pre_stimulus_onset * 1000.0)))

                eeg_data = []
                eeg_timestamps = []

                # `marker_data` is an array of shape [2, n_timepoints], where the [0, :]
                # corresponds to timestamps, and [1, :] corresponds to the marker values
                # (represented as a nested list here).
                marker_data = np.zeros((2, (images_per_block * 2)))

                # Image loop (within a block).
                for image_count in range(images_per_block):
                    if not running:  # Check for quit event
                        break

                    # ------------------------------------------------------------------
                    # *** Pre-stimulus interval

                    # Start of the pre-stimulus interval.
                    t_pre_stim_start = time()

                    img_rect = current_image.get_rect(
                        center=(screen_width // 2, screen_height // 2)
                    )
                    screen.fill(global_config.rest_condition_color)
                    screen.blit(current_image, img_rect)

                    # Wait until the end of the pre-stimulus period.
                    t_pre_stim_end = t_pre_stim_start + pre_stimulus_interval
                    while time() < t_pre_stim_end:
                        for event in pygame.event.get():
                            if event.type == pygame.QUIT:
                                running = False
                            if event.type == pygame.KEYDOWN:
                                if event.key == pygame.K_ESCAPE:
                                    running = False
                        if not running:
                            break
                    if not running:
                        break

                    # ------------------------------------------------------------------
                    # *** Stimulus period

                    pygame.display.flip()
                    t_stim_start = time()  # Start of stimulus presentation.

                    # Insert stimulus start marker and get its timestamp.
                    marker_val_stim_start, marker_ts_stim_start = (
                        eeg_device.insert_marker(global_config.stim_start_marker)
                    )
                    if marker_val_stim_start is not None:
                        data_logging_queue.put(
                            {
                                "type": "marker",
                                "marker_value": marker_val_stim_start,
                                "timestamp": marker_ts_stim_start,
                            }
                        )
                    marker_data[0, (image_count * 2)] = marker_ts_stim_start
                    marker_data[1, (image_count * 2)] = marker_val_stim_start

                    # Send pre-stimulus EEG data (to avoid buffer overflow).
                    eeg_data_pre_stim, eeg_ts_pre_stim = eeg_device.get_board_data()
                    if eeg_data_pre_stim.size > 0:
                        data_logging_queue.put(
                            {
                                "type": "eeg",
                                "eeg_data": eeg_data_pre_stim,
                                "eeg_timestamps": eeg_ts_pre_stim,
                            }
                        )
                    eeg_data.append(eeg_data_pre_stim)
                    eeg_timestamps.append(eeg_ts_pre_stim)

                    # Wait for image duration, but check for responses continuously.
                    t_stim_end_expected = t_stim_start + image_duration
                    while time() < t_stim_end_expected:
                        for event in pygame.event.get():
                            if event.type == pygame.QUIT:
                                running = False
                            if event.type == pygame.KEYDOWN:
                                if event.key == pygame.K_ESCAPE:
                                    running = False
                        if not running:
                            break
                    if not running:
                        break

                    # ------------------------------------------------------------------
                    # *** Post-stimulus period

                    # End of stimulus presentation. Display grey screen.
                    screen.fill(global_config.rest_condition_color)
                    pygame.display.flip()
                    t_stim_end_actual = time()

                    marker_val_stim_end, marker_ts_stim_end = eeg_device.insert_marker(
                        global_config.stim_end_marker
                    )
                    if marker_val_stim_end is not None:
                        data_logging_queue.put(
                            {
                                "type": "marker",
                                "marker_value": marker_val_stim_end,
                                "timestamp": marker_ts_stim_end,
                            }
                        )
                    marker_data[0, ((image_count * 2) + 1)] = marker_ts_stim_end
                    marker_data[1, ((image_count * 2) + 1)] = marker_val_stim_end

                    eeg_data_stim, eeg_ts_stim = eeg_device.get_board_data()
                    if eeg_data_stim.size > 0:
                        data_logging_queue.put(
                            {
                                "type": "eeg",
                                "eeg_data": eeg_data_stim,
                                "eeg_timestamps": eeg_ts_stim,
                            }
                        )
                    eeg_data.append(eeg_data_stim)
                    eeg_timestamps.append(eeg_ts_stim)

                    # Wait until the end of the post-stimulus period
                    t_post_stim_end = t_stim_end_actual + post_stimulus_interval
                    while time() < t_post_stim_end:
                        for event in pygame.event.get():
                            if event.type == pygame.QUIT:
                                running = False
                            if event.type == pygame.KEYDOWN:
                                if event.key == pygame.K_ESCAPE:
                                    running = False
                        if not running:
                            break
                    if not running:
                        break

                    eeg_data_post_stim, eeg_ts_post_stim = eeg_device.get_board_data()
                    if eeg_data_post_stim.size > 0:
                        data_logging_queue.put(
                            {
                                "type": "eeg",
                                "eeg_data": eeg_data_post_stim,
                                "eeg_timestamps": eeg_ts_post_stim,
                            }
                        )
                    eeg_data.append(eeg_data_post_stim)
                    eeg_timestamps.append(eeg_ts_post_stim)

                    # ------------------------------------------------------------------
                    # *** Log stimulus metadata

                    stimulus_data = {
                        "stimulus_start_time": t_stim_start,
                        "stimulus_end_time": t_stim_end_actual,
                        "stimulus_duration_s": t_stim_end_actual - t_stim_start,
                        "image_file_path": next_image_file_path,
                        "image_category": next_image_category,
                        "is_target_event": False,  # Dummy values, TODO: data logging for eeg-to-image
                        "response_time_s": 0.0,  # Dummy values, TODO: data logging for eeg-to-image
                    }
                    data_logging_queue.put(
                        {"type": "stimulus", "stimulus_data": stimulus_data}
                    )

                # ----------------------------------------------------------------------
                # *** Inference

                eeg_data = np.concatenate(eeg_data, axis=1)
                eeg_timestamps = np.concatenate(eeg_timestamps)

                request_dict = {
                    "eeg_data": eeg_data.tolist(),
                    "eeg_timestamps": eeg_timestamps.tolist(),
                    "marker_data": marker_data.tolist(),
                    "utility_frequency": utility_frequency,
                    "eeg_channel_mapping": eeg_channel_mapping,
                }

                request_json = json.dumps(request_dict)

                # Wueue to receive images from the websocket thread.
                image_queue = queue.Queue()

                client_thread = threading.Thread(
                    target=websocket_client_thread,
                    args=(api_endpoint, request_json, image_queue),
                    daemon=True,
                )
                client_thread.start()

                # ----------------------------------------------------------------------
                # *** Show generated images as they arrive

                generated_image_surface = None
                path_image_out = None
                eeg_model_id = "unknown"
                is_first_message = True

                # Loop to display images as they are received from the thread.
                while True:
                    try:
                        # Check the queue for a new message (non-blocking).
                        message = image_queue.get_nowait()

                        if message is None:  # Error signal
                            print("Error receiving image from server.")
                            running = False
                            break

                        if is_first_message:
                            # The first message contains the model ID.
                            eeg_model_id = message.get("eeg_model_id", "unknown")
                            is_first_message = False
                            continue  # Wait for the next message which will be an image

                        # Decode the base64 image.
                        image_bytes = base64.b64decode(message["image_base64"])
                        image_file = io.BytesIO(image_bytes)

                        # Create a Pygame surface
                        generated_image_surface = pygame.image.load(
                            image_file
                        ).convert()

                        # Scale the image for display.
                        scaled_image_surface = scale_image_surface(
                            image_surface=generated_image_surface,  # TODO
                            screen_width=screen_width,
                            screen_height=screen_height,
                        )

                        # Display the new image.
                        img_rect = scaled_image_surface.get_rect(
                            center=(screen_width // 2, screen_height // 2)
                        )
                        screen.fill(global_config.rest_condition_color)
                        screen.blit(scaled_image_surface, img_rect)
                        pygame.display.flip()

                        # If this is the final, high-quality image, save it.
                        if message.get("step") == "final":
                            time_now = get_formatted_current_datetime()
                            true_image_category = next_image_category
                            path_image_out = os.path.join(
                                output_dir_images,
                                f"{eeg_model_id}_{time_now}_{true_image_category}.png",
                            )
                            with open(path_image_out, "wb") as f:
                                f.write(image_bytes)
                            break  # Exit the image receiving loop

                    except queue.Empty:
                        # No new image in the queue, just continue the loop
                        pass

                    # Keep Pygame responsive
                    for event in pygame.event.get():
                        if event.type == pygame.QUIT or (
                            event.type == pygame.KEYDOWN
                            and event.key == pygame.K_ESCAPE
                        ):
                            running = False
                            break
                    if not running:
                        break

                if not running:
                    break

                # ----------------------------------------------------------------------
                # *** Show the final generated image for a fixed duration

                # Show generated image for this amount of time.
                t_generated_img_end = time() + generated_image_duration

                generated_image_and_metadata = load_and_scale_image(
                    image_file_path=path_image_out,
                    screen_width=screen_width,
                    screen_height=screen_height,
                )

                generated_image = generated_image_and_metadata["image"]

                img_rect = current_image.get_rect(
                    center=(screen_width // 2, screen_height // 2)
                )
                screen.fill(global_config.rest_condition_color)
                screen.blit(generated_image, img_rect)
                pygame.display.flip()

                # Insert stimulus start marker and get its timestamp.
                generated_img_start_marker = 3.0  # Hardcoded TODO make config param
                marker_val_stim_start, marker_ts_stim_start = eeg_device.insert_marker(
                    generated_img_start_marker
                )
                if marker_val_stim_start is not None:
                    data_logging_queue.put(
                        {
                            "type": "marker",
                            "marker_value": marker_val_stim_start,
                            "timestamp": marker_ts_stim_start,
                        }
                    )

                while time() < t_generated_img_end:
                    for event in pygame.event.get():
                        if event.type == pygame.QUIT:
                            running = False
                        if event.type == pygame.KEYDOWN:
                            if event.key == pygame.K_ESCAPE:
                                running = False
                    if not running:
                        break
                if not running:
                    break

                # ----------------------------------------------------------------------
                # *** Log EEG data (to avoid buffer overflow)

                # Log data from the interval when waiting for inference results.

                eeg_data_wait, eeg_ts_wait = eeg_device.get_board_data()
                if eeg_data_wait.size > 0:
                    data_logging_queue.put(
                        {
                            "type": "eeg",
                            "eeg_data": eeg_data_wait,
                            "eeg_timestamps": eeg_ts_wait,
                        }
                    )

                # ----------------------------------------------------------------------
                # *** Grey screen (after generated image)

                # End of generated image presentation. Display grey screen.
                screen.fill(global_config.rest_condition_color)
                pygame.display.flip()

                # Insert stimulus start marker and get its timestamp.
                generated_img_end_marker = 4.0  # Hardcoded TODO make config param
                marker_val_stim_end, marker_ts_stim_end = eeg_device.insert_marker(
                    generated_img_end_marker
                )
                if marker_val_stim_end is not None:
                    data_logging_queue.put(
                        {
                            "type": "marker",
                            "marker_value": marker_val_stim_end,
                            "timestamp": marker_ts_stim_end,
                        }
                    )

                # ----------------------------------------------------------------------
                # *** Log EEG data (to avoid buffer overflow)

                eeg_data_gen_img, eeg_ts_gen_img = eeg_device.get_board_data()
                if eeg_data_gen_img.size > 0:
                    data_logging_queue.put(
                        {
                            "type": "eeg",
                            "eeg_data": eeg_data_gen_img,
                            "eeg_timestamps": eeg_ts_gen_img,
                        }
                    )

                if not running:
                    break

                # ----------------------------------------------------------------------
                # *** Prepare next block

                # Update tracking variables for the next loop iteration.
                previous_image_file_path = next_image_file_path

                # Inter-block grey screen.
                screen.fill(global_config.rest_condition_color)
                pygame.display.flip()
                remaining_wait = (
                    inter_block_grey_duration
                    - tone_pre_stimulus_onset
                    + np.random.uniform(low=0.0, high=isi_jitter)
                )
                pygame.time.delay(int(round(remaining_wait * 1000.0)))

                # Send post-stimulus EEG data (to avoid buffer overflow).
                eeg_data, eeg_ts = eeg_device.get_board_data()
                if eeg_data.size > 0:
                    data_logging_queue.put(
                        {"type": "eeg", "eeg_data": eeg_data, "eeg_timestamps": eeg_ts}
                    )

            # --------------------------------------------------------------------------
            # *** End of experiment

            running = False

            # Send final board data.
            eeg_data, eeg_ts = eeg_device.get_board_data()
            if eeg_data.size > 0:
                data_logging_queue.put(
                    {"type": "eeg", "eeg_data": eeg_data, "eeg_timestamps": eeg_ts}
                )

        except Exception as e:
            print(f"An error occurred during the experiment: {e}")
            print(traceback.format_exc())
            running = False
        finally:
            pygame.quit()
            print("Experiment closed.")

    eeg_device.stop_stream()
    eeg_device.release_session()

    print("Join process for sending data")
    data_logging_queue.put(None)
    logging_process.join()
