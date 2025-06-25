import multiprocessing as mp
import os
import random
from time import sleep, time

import pygame
from brainflow.board_shim import BoardIds, BoardShim, BrainFlowInputParams

from nubrain.experiment.data import eeg_data_logging
from nubrain.experiment.global_config import GlobalConfig
from nubrain.image.tools import get_all_image_paths, load_and_scale_image
from nubrain.misc.datetime import get_formatted_current_datetime


def experiment(config: dict):
    # ----------------------------------------------------------------------------------
    # *** Get config

    demo_mode = config["demo_mode"]

    subject_id = config["subject_id"]
    session_id = config["session_id"]

    output_directory = config["output_directory"]
    image_directory = config["image_directory"]

    eeg_channel_mapping = config["eeg_channel_mapping"]

    # bandstop_low_cutoff_freq = config["bandstop_low_cutoff_freq"]
    # bandstop_high_cutoff_freq = config["bandstop_high_cutoff_freq"]
    # bandstop_filter_order = config["bandstop_filter_order"]

    # bandpass_low_cutoff_freq = config["bandpass_low_cutoff_freq"]
    # bandpass_high_cutoff_freq = config["bandpass_high_cutoff_freq"]
    # bandpass_filter_order = config["bandpass_filter_order"]

    initial_rest_duration = config["initial_rest_duration"]
    image_duration = config["image_duration"]
    isi_duration = config["isi_duration"]
    inter_block_grey_duration = config["inter_block_grey_duration"]

    n_blocks = config["n_blocks"]
    images_per_block = config["images_per_block"]

    eeg_device_address = config["eeg_device_address"]

    global_config = GlobalConfig()

    # ----------------------------------------------------------------------------------
    # *** Test if output path exists

    if not os.path.isdir(output_directory):
        raise AssertionError(f"Target directory does not exist: {output_directory}")

    current_datetime = get_formatted_current_datetime()
    path_out_data = os.path.join(output_directory, f"eeg_session_{current_datetime}.h5")

    if os.path.isfile(path_out_data):
        raise AssertionError(f"Target file aready exists: {path_out_data}")

    # ----------------------------------------------------------------------------------
    # *** Get image paths

    image_file_paths = get_all_image_paths(image_directory=image_directory)

    if not image_file_paths:
        raise AssertionError(f"Found no images at {image_directory}")
    print(f"Found {len(image_file_paths)} images")

    # ----------------------------------------------------------------------------------
    # *** Prepare EEG measurement

    BoardShim.enable_dev_board_logger()

    if demo_mode:
        board_id = BoardIds.SYNTHETIC_BOARD.value
    else:
        board_id = BoardIds.CYTON_BOARD.value

    params = BrainFlowInputParams()
    params.serial_port = eeg_device_address
    board = BoardShim(board_id, params)

    eeg_board_description = BoardShim.get_board_descr(board_id)

    # Replace (wrong) default channel names from Cyton board description with channel
    # mapping from config.
    eeg_channel_idxs = sorted([int(x) for x in list(eeg_channel_mapping.keys())])
    eeg_channel_names = []
    for eeg_channel_idx in eeg_channel_idxs:
        eeg_channel_names.append(eeg_channel_mapping[str(eeg_channel_idx)])
    # For example: 'O1,O2,T3,T4,T5,T6,F3,F4'
    eeg_board_description["eeg_names"] = ",".join(eeg_channel_names)

    eeg_sampling_rate = int(eeg_board_description["sampling_rate"])
    eeg_channels = eeg_board_description["eeg_channels"]  # Get EEG channel indices
    marker_channel = eeg_board_description["marker_channel"]

    board.prepare_session()

    print(f"Board: {eeg_board_description['name']}")
    print(f"Sampling Rate: {eeg_sampling_rate} Hz")
    print(f"EEG Channels: {eeg_channels}")

    board.start_stream()

    sleep(0.1)
    board_data = board.get_board_data()

    print(f"Board data dtype: {board_data.dtype}")

    # Total number of channels, including EEG, marker, and other channels.
    n_channels_total = board_data.shape[0]

    # ----------------------------------------------------------------------------------
    # *** Start data logging subprocess

    data_logging_queue = mp.Queue()

    subprocess_params = {
        "demo_mode": demo_mode,
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
        # Misc
        # "nubrain_endpoint": nubrain_endpoint,
        # "nubrain_api_key": nubrain_api_key,
        "path_out_data": path_out_data,
        "data_logging_queue": data_logging_queue,
    }

    logging_process = mp.Process(
        target=eeg_data_logging,
        args=(subprocess_params,),
    )

    logging_process.Daemon = True
    logging_process.start()

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

        font = pygame.font.Font(None, 48)  # Basic font for messages

        # Load first image.
        image_and_metadata = None
        while image_and_metadata is None:
            # Select a random image from the full list.
            random_image_file_path = random.choice(image_file_paths)
            image_and_metadata = load_and_scale_image(
                image_file_path=random_image_file_path,
                screen_width=screen_width,
                screen_height=screen_height,
            )

        try:
            # Initial grey screen.
            pygame.time.wait(100)
            screen.fill(global_config.rest_condition_color)
            pygame.display.flip()
            pygame.time.wait(100)
            screen.fill(global_config.rest_condition_color)
            pygame.display.flip()

            # Clear board buffer.
            _ = board.get_board_data()

            # Pause for specified number of milliseconds.
            pygame.time.delay(int(round(initial_rest_duration * 1000.0)))

            # Block loop.
            for idx_block in range(n_blocks):
                print(f"Starting Block {idx_block + 1} out of {n_blocks}")

                # Image loop (within a block).
                for image_count in range(images_per_block):
                    if not running:
                        break  # Check for quit event

                    image_file_path = image_and_metadata["image_file_path"]
                    current_image = image_and_metadata["image"]
                    image_category = image_and_metadata["image_category"]

                    img_rect = current_image.get_rect(
                        center=(screen_width // 2, screen_height // 2)
                    )

                    # Display image. Clear previous screen content.
                    screen.fill(global_config.rest_condition_color)
                    screen.blit(current_image, img_rect)
                    pygame.display.flip()

                    # Start of stimulus presentation.
                    t1 = time()
                    # Insert stimulus start maker into EEG data.
                    board.insert_marker(global_config.stim_start_marker)

                    # Send pre-stimulus board data (to avoid buffer overflow).
                    data_to_queue = {
                        "board_data": board.get_board_data(),
                        "stimulus_data": None,
                    }
                    data_logging_queue.put(data_to_queue)

                    # Time until when to show stimulus.
                    t2 = t1 + image_duration
                    while time() < t2:
                        pass

                    # End of stimulus presentation. Display ISI grey screen.
                    screen.fill(global_config.rest_condition_color)
                    pygame.display.flip()
                    board.insert_marker(global_config.stim_end_marker)
                    t3 = time()

                    # Send data corresponding to stimulus period.
                    stimulus_data = {
                        "stimulus_start_time": t1,
                        "stimulus_end_time": t3,
                        "stimulus_duration_s": t3 - t1,
                        "image_file_path": image_file_path,
                        "image_category": image_category,
                    }
                    data_to_queue = {
                        "board_data": board.get_board_data(),
                        "stimulus_data": stimulus_data,
                    }
                    data_logging_queue.put(data_to_queue)

                    # Load next image.
                    image_and_metadata = None
                    while image_and_metadata is None:
                        # Select a random image from the full list.
                        random_image_file_path = random.choice(image_file_paths)
                        image_and_metadata = load_and_scale_image(
                            image_file_path=random_image_file_path,
                            screen_width=screen_width,
                            screen_height=screen_height,
                        )

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

                # Send post-stimulus board data (to avoid buffer overflow).
                data_to_queue = {
                    "board_data": board.get_board_data(),
                    "stimulus_data": None,
                }
                data_logging_queue.put(data_to_queue)

                # Inter-block grey screen.
                print(f"End of Block {idx_block + 1}. Starting inter-block interval.")
                screen.fill(global_config.rest_condition_color)
                pygame.display.flip()
                # We already waited for the ISI duration, therefore subtract it from the
                # inter block duration. Avoid negative value is case ISI duration is
                # longer than inter block duration.
                remaining_wait = max((inter_block_grey_duration - isi_duration), 0.0)
                pygame.time.delay(int(round(remaining_wait * 1000.0)))

            # End of experiment.
            if running:  # Only show if not quit early
                screen.fill(global_config.rest_condition_color)
                end_text = font.render("Experiment complete.", True, (0.0, 0.0, 0.0))
                text_rect = end_text.get_rect(
                    center=(screen_width // 2, screen_height // 2)
                )
                screen.blit(end_text, text_rect)
                pygame.display.flip()
                pygame.time.wait(500)

            running = False

            # Send final board data.
            data_to_queue = {
                "board_data": board.get_board_data(),
                "stimulus_data": None,
            }
            data_logging_queue.put(data_to_queue)

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
    data_logging_queue.put(None)
    logging_process.join()
