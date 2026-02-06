import multiprocessing as mp
import os
import traceback
from time import sleep

import numpy as np
import pygame

from nubrain.audio.tone import generate_tone
from nubrain.device.device_interface import create_eeg_device
from nubrain.experiment_text.data import eeg_data_logging
from nubrain.experiment_text.random_target_events import sample_target_events
from nubrain.experiment_text.text_config import TextConfig
from nubrain.misc.datetime import get_formatted_current_datetime
from nubrain.text.tools import load_and_preprocess_text

mp.set_start_method("spawn", force=True)  # Necessary on if running on windows?


def experiment_text(config: dict):
    # ----------------------------------------------------------------------------------
    # *** Get config

    device_type = config["device_type"]
    lsl_stream_name = config.get("lsl_stream_name", "DSI-24")

    subject_id = config["subject_id"]
    session_id = config["session_id"]

    output_directory = config["output_directory"]
    path_text = config["path_text"]

    eeg_channel_mapping = config.get("eeg_channel_mapping", None)

    utility_frequency = config["utility_frequency"]

    initial_rest_duration = config["initial_rest_duration"]
    stimulus_duration = config["stimulus_duration"]
    isi_duration = config["isi_duration"]
    isi_jitter = config["isi_jitter"]
    isi_extension_target = config["isi_extension_target"]
    inter_block_rest_duration = config["inter_block_rest_duration"]
    response_window_duration = config["response_window_duration"]

    word_idx_start = config["word_idx_start"]
    n_words_to_show = config["n_words_to_show"]
    n_target_events = config["n_target_events"]
    min_distance_targets = config["min_distance_targets"]
    stimuli_per_block = config["stimuli_per_block"]
    stimulus_font_size = config["stimulus_font_size"]

    eeg_device_address = config.get("eeg_device_address", None)

    text_config = TextConfig()

    # ----------------------------------------------------------------------------------
    # *** Test if output path exists

    if not os.path.isdir(output_directory):
        raise AssertionError(f"Target directory does not exist: {output_directory}")

    current_datetime = get_formatted_current_datetime()
    path_out_data = os.path.join(output_directory, f"eeg_session_{current_datetime}.h5")

    if os.path.isfile(path_out_data):
        raise AssertionError(f"Target file already exists: {path_out_data}")

    # ----------------------------------------------------------------------------------
    # *** Load text

    # Load text from file.
    text = load_and_preprocess_text(path_text=path_text)

    # Select subset of text.
    text = text[word_idx_start : (word_idx_start + n_words_to_show)]

    # Random target events. In case of a target event, the word will be repeated.
    text_and_targets = sample_target_events(
        text=text,
        n_target_events=n_target_events,
        min_distance_targets=min_distance_targets,
    )

    text = text_and_targets["text_with_targets"]
    is_target = text_and_targets["is_target"]

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
        "path_text": path_text,
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
        "stimulus_duration": stimulus_duration,
        "isi_duration": isi_duration,
        "isi_jitter": isi_jitter,
        "isi_extension_target": isi_extension_target,
        "inter_block_rest_duration": inter_block_rest_duration,
        "response_window_duration": response_window_duration,
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
        "path_out_data": path_out_data,
        "data_logging_queue": data_logging_queue,
    }

    logging_process = mp.Process(target=eeg_data_logging, args=(subprocess_params,))
    logging_process.daemon = True
    logging_process.start()

    # ----------------------------------------------------------------------------------
    # *** Start experiment

    # Performance counters.
    n_hits = 0
    n_false_alarms = 0
    n_total_targets = sum(is_target)

    running = True
    while running:
        pygame.init()

        # ------------------------------------------------------------------------------
        # *** Prepare audio cues

        # Use an audio cue at the beginning and at the end of the inter-block interval,
        # so the participant can close their eyes / rest.

        # How long before the end of the inter-block interval to play the audio cue.
        pure_tone_end_delay = 1.0

        # Play the tone to cue the end of the inter-block interval x seconds before the
        # end of the inter-block interval.
        # Do not use the audio cue if the inter-block interval is too short.
        if inter_block_rest_duration <= (pure_tone_end_delay + 0.1):
            print(
                "WARNING: Will not use audio cue for the end of the inter-block "
                "interval because of short inter-block interval of "
                f"{inter_block_rest_duration} s"
            )
            use_ibi_audio_cue = False
        else:
            use_ibi_audio_cue = True

        if use_ibi_audio_cue:
            pygame.mixer.init(frequency=44100, size=-16, channels=2, buffer=512)

            # Get the sample rate from the mixer settings.
            sample_rate = pygame.mixer.get_init()[0]

            # Generate the tone data.
            tone_data_start = generate_tone(
                frequency=700,  # Pitch of the tone in Hz
                duration=0.3,  # Duration of audio cue
                amplitude=0.9,  # Volume, from 0.0 to 1.0
                sample_rate=sample_rate,
            )

            tone_data_end = generate_tone(
                frequency=1400,  # Pitch of the tone in Hz
                duration=0.3,  # Duration of audio cue
                amplitude=0.9,  # Volume, from 0.0 to 1.0
                sample_rate=sample_rate,
            )

            # Create a sound object from the numpy array.
            pure_tone_start = pygame.sndarray.make_sound(tone_data_start)
            pure_tone_end = pygame.sndarray.make_sound(tone_data_end)

        # ------------------------------------------------------------------------------
        # *** Prepare visual stimulus generation

        # Get screen dimensions and set up full screen.
        screen_info = pygame.display.Info()
        screen_width = screen_info.current_w
        screen_height = screen_info.current_h
        screen = pygame.display.set_mode(
            (screen_width, screen_height), pygame.FULLSCREEN
        )
        pygame.display.set_caption("Silent Reading Experiment")
        pygame.mouse.set_visible(False)

        try:
            # Initial grey screen.
            pygame.time.wait(100)
            screen.fill(text_config.rest_condition_color)
            pygame.display.flip()
            pygame.time.wait(100)
            screen.fill(text_config.rest_condition_color)
            pygame.display.flip()

            stimulus_font = pygame.font.SysFont("timesnewroman", stimulus_font_size)

            # Clear board buffer.
            _, _ = eeg_device.get_board_data()

            # Pause for specified number of milliseconds.
            pygame.time.delay(int(round(initial_rest_duration * 1000.0)))

            # Send pre-stimulus EEG data (to avoid buffer overflow).
            eeg_data, eeg_ts = eeg_device.get_board_data()
            if eeg_data.size > 0:
                data_logging_queue.put(
                    {
                        "type": "eeg",
                        "eeg_data": eeg_data,
                        "eeg_timestamps": eeg_ts,
                    }
                )

            # Count stimuli to introduce breaks between blocks.
            stimulus_block_counter = 0

            # Loop through words.
            for word, is_target_event in zip(text, is_target):
                if not running:  # Check for quit event
                    break

                stimulus_text = stimulus_font.render(
                    word,
                    True,
                    text_config.font_color,
                )
                stimulus_rect = stimulus_text.get_rect(
                    center=(screen_width // 2, screen_height // 2)
                )
                screen.blit(stimulus_text, stimulus_rect)
                pygame.display.flip()
                # Start of stimulus presentation.
                t_stim_start = eeg_device.lsl_local_clock()

                # When using an OpenBCI device, we insert a stimulus marker into the
                # time series data on the EEG board. These markers can be used during
                # analysis to identify stimulus events. For the DSI-24 device, we
                # instead use LSL timestamps stored in the hdf5 file for identifying
                # stimulus events.
                if device_type in ["cyton", "synthetic"]:
                    eeg_device.insert_marker(text_config.stim_start_marker)
                else:
                    data_logging_queue.put(
                        {
                            "type": "marker",
                            "marker_value": text_config.stim_start_marker,
                            "timestamp": t_stim_start,
                        }
                    )

                response_made = False
                response_time = np.nan
                response_deadline = t_stim_start + response_window_duration

                # Wait for stimulus duration, but check for responses continuously.
                t_stim_end_expected = t_stim_start + stimulus_duration
                while eeg_device.lsl_local_clock() < t_stim_end_expected:
                    for event in pygame.event.get():
                        if event.type == pygame.QUIT:
                            running = False
                        if event.type == pygame.KEYDOWN:
                            keydown_time = eeg_device.lsl_local_clock()
                            if event.key == pygame.K_ESCAPE:
                                running = False
                            # Check for space bar press within the response window.
                            if event.key == pygame.K_SPACE and not response_made:
                                if keydown_time < response_deadline:
                                    response_made = True
                                    response_time = keydown_time - t_stim_start
                                    print(f"Response time: {round(response_time, 3)}")
                                    if is_target_event:
                                        # Hit.
                                        n_hits += 1
                                    else:
                                        # False alarm.
                                        n_false_alarms += 1
                    if not running:
                        break
                if not running:
                    break

                # End of stimulus presentation. Display ISI grey screen.
                screen.fill(text_config.rest_condition_color)
                pygame.display.flip()
                t_stim_end_actual = eeg_device.lsl_local_clock()

                # When using an OpenBCI device, we insert a stimulus marker into the
                # time series data on the EEG board. These markers can be used during
                # analysis to identify stimulus events. For the DSI-24 device, we
                # instead use LSL timestamps stored in the hdf5 file for identifying
                # stimulus events.
                if device_type in ["cyton", "synthetic"]:
                    eeg_device.insert_marker(text_config.stim_end_marker)
                else:
                    data_logging_queue.put(
                        {
                            "type": "marker",
                            "marker_value": text_config.stim_end_marker,
                            "timestamp": t_stim_end_actual,
                        }
                    )

                eeg_data, eeg_ts = eeg_device.get_board_data()
                if eeg_data.size > 0:
                    data_logging_queue.put(
                        {
                            "type": "eeg",
                            "eeg_data": eeg_data,
                            "eeg_timestamps": eeg_ts,
                        }
                    )

                # Time until when to show grey screen (ISI).
                t_isi_end = (
                    t_stim_end_actual
                    + isi_duration
                    + np.random.uniform(low=0.0, high=isi_jitter)
                )

                if is_target_event:
                    # If this is a target event, prolong the ISI duration, to allow the
                    # subject to respond before the onset of the next stimulus, to
                    # reduce the probability of a motor response artefact in the
                    # subsequent trial.
                    t_isi_end += isi_extension_target

                # Continue checking for late responses or quit events.
                while eeg_device.lsl_local_clock() < t_isi_end:
                    for event in pygame.event.get():
                        if event.type == pygame.QUIT:
                            running = False
                        if event.type == pygame.KEYDOWN:
                            keydown_time = eeg_device.lsl_local_clock()
                            if event.key == pygame.K_ESCAPE:
                                running = False
                            # Still check for spacebar presses that are within the
                            # response window for target events.
                            if event.key == pygame.K_SPACE and not response_made:
                                if keydown_time < response_deadline:
                                    response_made = True
                                    response_time = keydown_time - t_stim_start
                                    print(f"Response time: {round(response_time, 3)}")
                                    if is_target_event:
                                        # Hit.
                                        n_hits += 1
                                    else:
                                        # False alarm.
                                        n_false_alarms += 1
                    if not running:
                        break
                if not running:
                    break

                stimulus_data = {
                    "stimulus_start_time": t_stim_start,
                    "stimulus_end_time": t_stim_end_actual,
                    "stimulus_duration_s": t_stim_end_actual - t_stim_start,
                    "word": word,
                    "is_target_event": is_target_event,
                    "response_time_s": response_time,
                }
                data_logging_queue.put(
                    {"type": "stimulus", "stimulus_data": stimulus_data}
                )

                if not running:
                    break

                # Send post-stimulus EEG data (to avoid buffer overflow).
                eeg_data, eeg_ts = eeg_device.get_board_data()
                if eeg_data.size > 0:
                    data_logging_queue.put(
                        {"type": "eeg", "eeg_data": eeg_data, "eeg_timestamps": eeg_ts}
                    )

                stimulus_block_counter += 1

                if stimulus_block_counter == stimuli_per_block:
                    # Inter-block grey screen.
                    screen.fill(text_config.rest_condition_color)
                    pygame.display.flip()
                    # Start of inter-block interval.
                    t_ibi_start = eeg_device.lsl_local_clock()

                    if use_ibi_audio_cue:
                        # Audio cue to signal the beginning of the inter-block interval.
                        pure_tone_start.play()

                    # End of inter-block interval.
                    t_ibi_end = t_ibi_start + inter_block_rest_duration

                    if use_ibi_audio_cue:
                        # Time when to play audio cue to signal end of inter-block
                        # interval.
                        t_ibi_end_audio_cue = t_ibi_end - pure_tone_end_delay
                        ibi_end_cue_played_yet = False

                    while (
                        eeg_device.lsl_local_clock() < t_ibi_end
                    ):  # Continue inter-block interval?
                        if (
                            use_ibi_audio_cue
                            and (t_ibi_end_audio_cue <= eeg_device.lsl_local_clock())
                            and not ibi_end_cue_played_yet
                        ):  # Time to play end of inter-block interval cue?
                            # Play the cue to signal the end of the inter-block
                            # interval.
                            pure_tone_end.play()
                            ibi_end_cue_played_yet = True

                    stimulus_block_counter = 0

                    # Send inter-block EEG data (to avoid buffer overflow).
                    eeg_data, eeg_ts = eeg_device.get_board_data()
                    if eeg_data.size > 0:
                        data_logging_queue.put(
                            {
                                "type": "eeg",
                                "eeg_data": eeg_data,
                                "eeg_timestamps": eeg_ts,
                            }
                        )

            # End of word loop.

            # Calculate behavioural results.
            n_misses = n_total_targets - n_hits

            # Write behavioural results to hdf5 file.
            behavioural_data = {
                "n_total_targets": n_total_targets,
                "n_hits": n_hits,
                "n_misses": n_misses,
                "n_false_alarms": n_false_alarms,
            }
            data_logging_queue.put(
                {"type": "behavioural", "behavioural_data": behavioural_data}
            )

            if running:
                # Display behavioural results.
                screen.fill(text_config.rest_condition_color)

                # Behavioural results title.
                title_font = pygame.font.Font(None, 72)
                title_text = title_font.render(
                    "Experiment Complete",
                    True,
                    text_config.font_color,
                )
                title_rect = title_text.get_rect(
                    center=(screen_width // 2, screen_height // 2 - 150)
                )
                screen.blit(title_text, title_rect)

                # Behavioural results text.
                results_font = pygame.font.Font(None, 56)
                hits_text = results_font.render(
                    f"Hits: {n_hits}",
                    True,
                    text_config.font_color,
                )
                misses_text = results_font.render(
                    f"Misses: {n_misses}",
                    True,
                    text_config.font_color,
                )
                false_alarms_text = results_font.render(
                    f"False Alarms: {n_false_alarms}",
                    True,
                    text_config.font_color,
                )

                # Position and display results
                hits_rect = hits_text.get_rect(
                    center=(screen_width // 2, screen_height // 2 - 20)
                )
                misses_rect = misses_text.get_rect(
                    center=(screen_width // 2, screen_height // 2 + 40)
                )
                false_alarms_rect = false_alarms_text.get_rect(
                    center=(screen_width // 2, screen_height // 2 + 100)
                )

                screen.blit(hits_text, hits_rect)
                screen.blit(misses_text, misses_rect)
                screen.blit(false_alarms_text, false_alarms_rect)

                pygame.display.flip()
                pygame.time.wait(5000)  # Show results for 5 seconds

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
