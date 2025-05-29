import base64
import datetime
import glob
import io
import json
import multiprocessing as mp
import os
import pickle
import random
from time import time

import numpy as np
import pygame
import requests
from brainflow.board_shim import BoardIds, BoardShim, BrainFlowInputParams
from brainflow.data_filter import (
    DataFilter,
    DetrendOperations,
    FilterTypes,
)
from PIL import Image
from scipy.stats import zscore

# --------------------------------------------------------------------------------------
# *** Helper functions


def load_and_scale_images(*, image_directory, screen_width, screen_height):
    """
    Loads all PNG and JPEG images from a directory and scales them to fit the screen.
    """
    extensions = ("*.png", "*.jpg", "*.jpeg")
    image_files = []
    for ext in extensions:
        image_files.extend(glob.glob(os.path.join(image_directory, ext)))

    loaded_images = []
    if not image_files:
        print(f"No images found in directory: {image_directory}")
        return []

    print(f"Found {len(image_files)} images.")

    for filepath in image_files:
        try:
            img = pygame.image.load(filepath)
            img_rect = img.get_rect()

            # Calculate scaling factor to fit screen while maintaining aspect ratio
            scale_w = screen_width / img_rect.width
            scale_h = screen_height / img_rect.height
            scale = min(scale_w, scale_h)

            new_width = int(img_rect.width * scale)
            new_height = int(img_rect.height * scale)

            scaled_img = pygame.transform.smoothscale(img, (new_width, new_height))
            loaded_images.append({"image_filepath": filepath, "image": scaled_img})
            print(f"Loaded and scaled: {filepath}")
        except pygame.error as e:
            print(f"Error loading or scaling image {filepath}: {e}")
    return loaded_images


def resize_image(*, image_bytes, return_image_file_extension=False):
    """
    Resize image to maximal size before sending over network.
    """
    image = Image.open(io.BytesIO(image_bytes))

    if image.format == "JPEG":
        image_format = "JPEG"
        image_file_extension = ".jpg"
    elif image.format == "PNG":
        image_format = "PNG"
        image_file_extension = ".png"
    elif image.format == "WEBP":
        image_format = "WEBP"
        image_file_extension = ".webp"
    else:
        print(f"Unexpected image format, will use png: {image.format}")
        image_format = "PNG"
        image_file_extension = ".png"

    width, height = image.size

    # Maximum dimension.
    max_dimension = 256

    # Check if resizing is needed.
    if (width > max_dimension) or (height > max_dimension):
        # Calculate the new size maintaining the aspect ratio.
        if width > height:
            new_width = max_dimension
            new_height = int(max_dimension * height / width)
        else:
            new_height = max_dimension
            new_width = int(max_dimension * width / height)

        # Resize the image.
        image = image.resize((new_width, new_height), Image.Resampling.BILINEAR)

    # Convert the image to bytes.
    img_byte_arr = io.BytesIO()

    image.save(img_byte_arr, format=image_format)
    img_byte_arr = bytearray(img_byte_arr.getvalue())

    if return_image_file_extension:
        return img_byte_arr, image_file_extension
    else:
        return img_byte_arr


def get_formatted_current_datetime():
    """
    Returns the current date and time formatted as "YYYY-MM-DD-HHMMSS".

    Returns:
      str: The formatted date and time string.
    """
    now = datetime.datetime.now()
    formatted_datetime = now.strftime("%Y-%m-%d-%H%M%S")
    return formatted_datetime


# --------------------------------------------------------------------------------------
# *** Preprocessing & saving data


def preprocess(subprocess_params: dict):
    """
    Preprocess data, send data to remote endpoint, and save copy locally. To be run in
    separate process (using multiprocessing).
    """
    eeg_board_description = subprocess_params["eeg_board_description"]
    eeg_sampling_rate = subprocess_params["eeg_sampling_rate"]
    eeg_channels = subprocess_params["eeg_channels"]
    eeg_channel_mapping = subprocess_params["eeg_channel_mapping"]

    marker_channel = subprocess_params["marker_channel"]

    bandstop_low_cutoff_freq = subprocess_params["bandstop_low_cutoff_freq"]
    bandstop_high_cutoff_freq = subprocess_params["bandstop_high_cutoff_freq"]
    bandstop_filter_order = subprocess_params["bandstop_filter_order"]

    bandpass_low_cutoff_freq = subprocess_params["bandpass_low_cutoff_freq"]
    bandpass_high_cutoff_freq = subprocess_params["bandpass_high_cutoff_freq"]
    bandpass_filter_order = subprocess_params["bandpass_filter_order"]

    nubrain_endpoint = subprocess_params["nubrain_endpoint"]
    nubrain_api_key = subprocess_params["nubrain_api_key"]

    path_out_data = subprocess_params["path_out_data"]

    data_to_remote_queue = subprocess_params["data_to_remote_queue"]

    experiment_data = {
        "metadata": {
            "eeg_board_description": eeg_board_description,
            "eeg_sampling_rate": eeg_sampling_rate,
            "eeg_channels": eeg_channels,
            "eeg_channel_mapping": eeg_channel_mapping,
            "bandstop_low_cutoff_freq": bandstop_low_cutoff_freq,
            "bandstop_high_cutoff_freq": bandstop_high_cutoff_freq,
            "bandstop_filter_order": bandstop_filter_order,
            "bandpass_low_cutoff_freq": bandpass_low_cutoff_freq,
            "bandpass_high_cutoff_freq": bandpass_high_cutoff_freq,
            "bandpass_filter_order": bandpass_filter_order,
            "time": time(),
        },
        "data": [],
    }

    counter = 0

    while True:
        data_to_send = data_to_remote_queue.get(block=True)

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
        # *** Resize image

        image = Image.open(image_filepath)

        # Create a byte stream in memory.
        byte_arr = io.BytesIO()

        image.save(byte_arr, format="PNG")

        # Get the byte value from the byte stream
        image_bytes = byte_arr.getvalue()

        # Limit image size.
        current_image_bytes = resize_image(image_bytes=image_bytes)
        # Encode in base64.
        img_b64 = base64.b64encode(current_image_bytes).decode("utf-8")

        # ------------------------------------------------------------------------------
        # *** Filter

        # When using brainflow, we can't filter a batch of all channels at once. Perhaps
        # refactor and use scipy. Only filter EEG channels (not auxiliarry channels).
        for idx_channel in eeg_channels:
            # Apply detrending first (optional but often good practice for IIR filters)
            DataFilter.detrend(board_data[idx_channel], DetrendOperations.LINEAR.value)
            # Apply bandstop filter.
            DataFilter.perform_bandstop(
                data=board_data[idx_channel],
                sampling_rate=eeg_sampling_rate,
                start_freq=bandstop_low_cutoff_freq,
                stop_freq=bandstop_high_cutoff_freq,
                order=bandstop_filter_order,
                filter_type=FilterTypes.BUTTERWORTH.value,
                ripple=0.0,  # Ignored for Butterworth
            )
            # Apply bandpass filter.
            # For FilterTypes.BUTTERWORTH, the ripple parameter is ignored (can be 0.0)
            DataFilter.perform_bandpass(
                data=board_data[idx_channel],
                sampling_rate=eeg_sampling_rate,
                start_freq=bandpass_low_cutoff_freq,
                stop_freq=bandpass_high_cutoff_freq,
                order=bandpass_filter_order,
                filter_type=FilterTypes.BUTTERWORTH.value,
                ripple=0.0,  # Ripple is for Chebyshev filters, ignored for Butterworth
            )

        # ------------------------------------------------------------------------------
        # *** Truncate EEG data stream by start and end marker

        # The beginning of the stimulus presentation is marked by a start maker (-1.0).
        # The end is potentially also marked (-2.0). We assume that the marker signal is
        # in the last channel.
        marker_channel_data = board_data[marker_channel]

        start_idx = np.where(marker_channel_data == -1.0)[0]
        if start_idx.size != 0:
            start_idx = start_idx[0]
            board_data = board_data[:, start_idx:]

        end_idx = np.where(marker_channel_data == -2.0)[0]
        if end_idx.size != 0:
            end_idx = end_idx[0]
            board_data = board_data[:, :end_idx]

        # TODO: Send auxilliariy data (non-eeg-channels)?
        eeg_data = board_data[eeg_channels, :]

        # ------------------------------------------------------------------------------
        # *** Normalize

        eeg_data = zscore(eeg_data, axis=1)

        # ------------------------------------------------------------------------------
        # *** Truncate data

        print(f"eeg_data: {eeg_data.shape}")

        # TODO: Fix hardcoded time interval.
        truncated_eeg_data = eeg_data[:, 5:]
        truncated_eeg_data = truncated_eeg_data[:, :110]

        print(f"truncated_eeg_data: {truncated_eeg_data.shape}")

        # ------------------------------------------------------------------------------
        # *** Send to remote endpoint

        # TODO: Add the following metadata before sending?
        # "eeg_board_description": eeg_board_description,
        # "eeg_sampling_rate": eeg_sampling_rate,
        # "eeg_channels": eeg_channels,
        # "eeg_channel_mapping": eeg_channel_mapping,

        payload = {
            "eeg": truncated_eeg_data.tolist(),
            "image": "data:image/jpeg;base64," + img_b64,
            "metadata": metadata,
        }

        payload = json.dumps(payload)

        response = requests.post(
            url=nubrain_endpoint,
            data=payload,
            headers={"Authorization": f"Bearer {nubrain_api_key}"},
        )

        if response.status_code != 200:
            print(
                f"Failed to send data. Status code: {response.status_code}. "
                + f"Response: {response.text}"
            )

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


def experiment(
    nubrain_api_key: str,
    eeg_channel_mapping: dict = {
        0: "O1",  # N1 Cyton pin
        1: "O2",  # N2 Cyton pin
        2: "P4",  # N3 Cyton pin
        3: "P3",  # N4 Cyton pin
        4: "C4",  # N5 Cyton pin
        5: "C3",  # N6 Cyton pin
        6: "F4",  # N7 Cyton pin
        7: "F3",  # N8 Cyton pin
    },
):
    # ----------------------------------------------------------------------------------
    # *** Test if path exists

    if not os.path.isdir(output_directory):
        raise AssertionError(f"Target directory does not exist: {output_directory}")

    current_datetime = get_formatted_current_datetime()
    path_out_data = os.path.join(
        output_directory, f"eeg_session_{current_datetime}.pickle"
    )

    if os.path.isfile(path_out_data):
        raise AssertionError(f"Target file aready exists: {path_out_data}")

    # ----------------------------------------------------------------------------------
    # *** Prepare EEG measurement

    BoardShim.enable_dev_board_logger()

    if demo_mode:
        board_id = BoardIds.SYNTHETIC_BOARD.value
    else:
        board_id = BoardIds.CYTON_BOARD.value

    params = BrainFlowInputParams()
    params.serial_port = "/dev/ttyUSB0"  # ?
    board = BoardShim(board_id, params)

    eeg_board_description = BoardShim.get_board_descr(board_id)
    eeg_sampling_rate = int(eeg_board_description["sampling_rate"])
    eeg_channels = eeg_board_description["eeg_channels"]  # Get EEG channel indices
    marker_channel = eeg_board_description["marker_channel"]

    board.prepare_session()

    print(f"Board: {eeg_board_description['name']}")
    print(f"Sampling Rate: {eeg_sampling_rate} Hz")
    print(f"EEG Channels: {eeg_channels}")

    board.start_stream()

    # ----------------------------------------------------------------------------------
    # *** Start preprocessing subprocess

    data_to_remote_queue = mp.Queue()

    subprocess_params = {
        "eeg_board_description": eeg_board_description,
        "eeg_sampling_rate": eeg_sampling_rate,
        "eeg_channels": eeg_channels,
        "eeg_channel_mapping": eeg_channel_mapping,
        "marker_channel": marker_channel,
        "bandstop_low_cutoff_freq": bandstop_low_cutoff_freq,
        "bandstop_high_cutoff_freq": bandstop_high_cutoff_freq,
        "bandstop_filter_order": bandstop_filter_order,
        "bandpass_low_cutoff_freq": bandpass_low_cutoff_freq,
        "bandpass_high_cutoff_freq": bandpass_high_cutoff_freq,
        "bandpass_filter_order": bandpass_filter_order,
        "nubrain_endpoint": nubrain_endpoint,
        "nubrain_api_key": nubrain_api_key,
        "path_out_data": path_out_data,
        "data_to_remote_queue": data_to_remote_queue,
    }

    preprocess_process = mp.Process(
        target=preprocess,
        args=(subprocess_params,),
    )

    preprocess_process.Daemon = True
    preprocess_process.start()

    # ----------------------------------------------------------------------------------
    # *** Start experiment

    running = True
    while running:
        pygame.init()

        # Get screen dimensions and set up full screen
        screen_info = pygame.display.Info()
        screen_width = screen_info.current_w
        screen_height = screen_info.current_h
        screen = pygame.display.set_mode(
            (screen_width, screen_height), pygame.FULLSCREEN
        )
        pygame.display.set_caption("Image Presentation Experiment")
        pygame.mouse.set_visible(False)  # Hide the mouse cursor

        # Load images
        all_images = load_and_scale_images(
            image_directory=image_directory,
            screen_width=screen_width,
            screen_height=screen_height,
        )

        if not all_images:
            print("No images loaded. Exiting.")
            pygame.quit()
            break

        font = pygame.font.Font(None, 48)  # Basic font for messages

        try:
            # 1. Initial grey screen
            print("Starting initial grey screen...")
            # sleep(0.1)
            pygame.time.wait(100)
            screen.fill(GREY)
            pygame.display.flip()
            # sleep(0.1)
            pygame.time.wait(100)
            screen.fill(GREY)
            pygame.display.flip()
            # sleep(initial_grey_duration)
            pygame.time.wait(int(round(initial_grey_duration * 100.0)))

            # Block loop.
            for block_num in range(n_blocks):
                print(f"\nStarting Block {block_num + 1}/{n_blocks}")

                # Image loop (within a block).
                for image_count in range(images_per_block):
                    if not running:
                        break  # Check for quit event

                    # Select a random image from the full list.
                    random_sample = random.choice(all_images)
                    image_filepath = random_sample["image_filepath"]
                    current_image = random_sample["image"]

                    img_rect = current_image.get_rect(
                        center=(screen_width // 2, screen_height // 2)
                    )

                    # Clear board buffer
                    _ = board.get_board_data()

                    # Display image. Clear previous screen content (optional, good practice).
                    screen.fill(GREY)
                    screen.blit(current_image, img_rect)
                    pygame.display.flip()

                    # Start of stimulus presentation.
                    t1 = time()
                    # Insert stimulus start maker into EEG data.
                    board.insert_marker(stim_start_marker)

                    # Time until when to show stimulus.
                    t2 = t1 + image_duration
                    while time() < t2:
                        pass

                    # End of stimulus presentation. Display ISI grey screen.
                    screen.fill(GREY)
                    pygame.display.flip()
                    board.insert_marker(stim_end_marker)
                    t3 = time()

                    board_data = (
                        board.get_board_data()
                    )  # Gets all data accumulated so far

                    metadata = {
                        "stimulus_start_time": t1,
                        "stimulus_end_time": t3,
                        "stimulus_duration_s": t3 - t1,
                    }

                    data_to_send = {
                        "board_data": board_data,
                        "metadata": metadata,
                        "image_filepath": image_filepath,
                    }

                    data_to_remote_queue.put(data_to_send)

                    # Time until when to show grey screen.
                    t4 = t3 + isi_duration
                    while time() < t4:
                        pass

                    # Event handling (allow quitting with ESC or window close).
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

                # 2. Inter-block grey screen (if not the last block)
                if block_num < n_blocks - 1:
                    print(
                        f"End of Block {block_num + 1}. Starting inter-block grey screen..."
                    )
                    screen.fill(GREY)
                    pygame.display.flip()
                    # sleep(inter_block_grey_duration)
                    pygame.time.wait(int(round(inter_block_grey_duration * 100.0)))
                else:
                    print(f"End of Block {block_num + 1}. Experiment finished.")

            # Final message (optional)
            if running:  # Only show if not quit early
                screen.fill(GREY)
                end_text = font.render("Experiment Complete.", True, BLACK)
                text_rect = end_text.get_rect(
                    center=(screen_width // 2, screen_height // 2)
                )
                screen.blit(end_text, text_rect)
                pygame.display.flip()
                # sleep(0.5)
                pygame.time.wait(500)

            running = False

        except Exception as e:
            print(f"An error occurred during the experiment: {e}")
            running = False
        finally:
            pygame.quit()
            print("Experiment closed.")

    board.stop_stream()
    board.release_session()

    # Join process for sending data.
    print("Join process for sending data")
    data_to_remote_queue.put(None)
    preprocess_process.join()
