"""
Data collection mode with text stimuli. Multiple choice comprehension questions at the
end of the run to ensure participant's attention.
"""

import json
import multiprocessing as mp
import os
import random
import traceback
from copy import deepcopy
from time import sleep

import numpy as np
import pygame

from nubrain.audio.tone import generate_tone
from nubrain.device.device_interface import create_eeg_device
from nubrain.experiment_text_comprehension.data import eeg_data_logging
from nubrain.experiment_text_comprehension.wrap_text import draw_text_wrapped
from nubrain.experiment_text_targets.text_config import TextConfig
from nubrain.misc.datetime import get_formatted_current_datetime
from nubrain.text.rendering import construct_fonts, render_spaced_text

mp.set_start_method("spawn", force=True)  # Necessary on if running on windows?


def experiment_text_comprehension(config: dict):
    # ----------------------------------------------------------------------------------
    # *** Get config

    device_type = config["device_type"]
    lsl_stream_name = config.get("lsl_stream_name", "DSI-24")

    subject_id = config["subject_id"]
    session_id = config["session_id"]

    output_directory = config["output_directory"]
    path_stimuli = config["path_stimuli"]

    storage_bucket_name = config["storage_bucket_name"]
    storage_blob_name = config["storage_blob_name"]
    storage_bucket_credentials = config["storage_bucket_credentials"]

    eeg_channel_mapping = config.get("eeg_channel_mapping", None)

    utility_frequency = config["utility_frequency"]

    initial_rest_duration = config["initial_rest_duration"]
    stimulus_duration = config["stimulus_duration"]
    stimulus_jitter = config["stimulus_jitter"]
    isi_duration = config["isi_duration"]
    isi_jitter = config["isi_jitter"]
    inter_block_rest_duration = config["inter_block_rest_duration"]
    n_chars_long_word_threshold = config["n_chars_long_word_threshold"]
    extra_duration_per_char = config["extra_duration_per_char"]
    max_extra_stimulus_duration = config["max_extra_stimulus_duration"]

    section_idx_start = config["section_idx_start"]

    stimuli_per_block = config["stimuli_per_block"]
    stimulus_font_sizes = config["stimulus_font_sizes"]
    stimulus_font_min_spacing = config["stimulus_font_min_spacing"]
    stimulus_font_max_spacing = config["stimulus_font_max_spacing"]

    eeg_device_address = config.get("eeg_device_address", None)

    text_config = TextConfig()

    # ----------------------------------------------------------------------------------
    # *** Test if output path exists

    if not os.path.isdir(output_directory):
        raise AssertionError(f"Target directory does not exist: {output_directory}")

    current_datetime = get_formatted_current_datetime()
    path_out_data = os.path.join(output_directory, f"eeg_{current_datetime}.h5")

    if os.path.isfile(path_out_data):
        raise AssertionError(f"Target file already exists: {path_out_data}")

    # ----------------------------------------------------------------------------------
    # *** Load stimulus data from JSON file

    with open(path_stimuli, "r", encoding="utf-8") as file:
        stimuli = json.load(file)

    # Only used for logging.
    words_per_section = stimuli["words_per_section"]
    min_words_per_section = stimuli["min_words_per_section"]
    n_answers = stimuli["n_answers"]  # Answer options per question
    n_questions = stimuli["n_questions"]

    stimuli = stimuli["stimulus_data"]
    # `stimuli` is a list of dictionaries:
    # stimuli = [{"text_section": "...", "questions_and_answers": "..."}, ...]

    # Select section for current run. In this condition (with comprehension questions)
    # we show only one section per run.
    stimuli = stimuli[section_idx_start]

    text = stimuli["text_section"]
    questions_and_answers = stimuli["questions_and_answers"]

    # Split text into individual words.
    text = text.split(" ")

    # Remove empty strings.
    text = [x for x in text if len(x) > 0]

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
        "path_stimuli": path_stimuli,
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
        "inter_block_rest_duration": inter_block_rest_duration,
        "n_chars_long_word_threshold": n_chars_long_word_threshold,
        "extra_duration_per_char": extra_duration_per_char,
        "max_extra_stimulus_duration": max_extra_stimulus_duration,
        # Experiment structure
        "section_idx_start": section_idx_start,
        "min_words_per_section": min_words_per_section,
        "words_per_section": words_per_section,
        "stimuli_per_block": stimuli_per_block,
        "stimulus_font_sizes": stimulus_font_sizes,
        "stimulus_font_min_spacing": stimulus_font_min_spacing,
        "stimulus_font_max_spacing": stimulus_font_max_spacing,
        # Text and targets
        "text": text,  # List of str
        "is_target": is_target,  # List of bool
        # Storage
        "path_out_data": path_out_data,
        "storage_bucket_name": storage_bucket_name,
        "storage_blob_name": storage_blob_name,
        "storage_bucket_credentials": storage_bucket_credentials,
        # Misc
        "utility_frequency": utility_frequency,
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
        # *** Prepare audio cues

        # Use an audio cue at the beginning and at the end of the inter-block interval,
        # so the participant can close their eyes / rest.

        # How long before the end of the inter-block interval to play the audio cue.
        pure_tone_end_delay = 1.0

        # Play the tone to cue the end of the inter-block interval x seconds before the
        # end of the inter-block interval. Do not use the audio cue if the inter-block
        # interval is too short.
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

            stimulus_fonts = construct_fonts(font_sizes=stimulus_font_sizes)

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
            t_stim_end_actual = None
            need_to_log_previous_stimulus = False

            # Loop through words.
            for word in text:
                if not running:  # Check for quit event
                    break

                # Extend stimulus duration for long words.
                n_chars = len(word)
                if n_chars > n_chars_long_word_threshold:
                    # By how many characters does the current word exceed the character
                    # threshold for extending stimulus duration.
                    n_excess_chars = n_chars - n_chars_long_word_threshold
                    extra_stimulus_duration = extra_duration_per_char * n_excess_chars
                    # Never prolong stimulus duration for more than x seconds
                    # (irrespective of number of characters).
                    extra_stimulus_duration = min(
                        extra_stimulus_duration, max_extra_stimulus_duration
                    )
                else:
                    # Word length is not above threshold (regular stimulus duration).
                    extra_stimulus_duration = 0.0

                # Randomly sample a font (we render the stimulus using different fonts
                # to achieve different stimulus appearance in terms of low-level visual
                # features.
                font_data = random.choice(stimulus_fonts)
                font_name = font_data["font_name"]
                font_size = font_data["font_size"]
                font_is_bold = font_data["font_is_bold"]
                font_is_italic = font_data["font_is_italic"]
                font_color = random.choice(text_config.font_colors)
                font_spacing = np.random.uniform(
                    low=stimulus_font_min_spacing,
                    high=stimulus_font_max_spacing,
                )

                # Clear previous stimulus.
                screen.fill(text_config.rest_condition_color)

                stimulus_text = render_spaced_text(
                    text=word,
                    font=font_data["font"],
                    color=font_color,
                    spacing=font_spacing,
                )

                stimulus_rect = stimulus_text.get_rect(
                    center=(screen_width // 2, screen_height // 2)
                )
                screen.blit(stimulus_text, stimulus_rect)

                # ----------------------------------------------------------------------
                # *** Stimulus

                pygame.display.flip()
                # Start of stimulus presentation.
                t_stim_start = eeg_device.lsl_local_clock()

                # ----------------------------------------------------------------------
                # *** Log previous stimulus

                # Now that we have flipped the screen and are showing the stimulus, take
                # the time and log data from previous stimulus. We do not need to log
                # the previous stimulus if there was an ISI or inter-block interval (in
                # that case, the stimulus gets logged at the beginning of the ISI or
                # inter-block interval, because the beginning of that interval
                # determines the stimulus end time).
                if need_to_log_previous_stimulus:
                    if t_stim_end_actual is None:
                        # If there was no ISI or inter-block interval, the end time of
                        # the previous stimulus is determined by the onset of the
                        # current stimulus.
                        t_stim_end_actual = t_stim_start

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

                    stimulus_data["stimulus_end_time"] = t_stim_end_actual
                    stimulus_data["stimulus_duration_s"] = (
                        t_stim_end_actual - stimulus_data["stimulus_start_time"]
                    )

                    data_logging_queue.put(
                        {"type": "stimulus", "stimulus_data": stimulus_data}
                    )

                # ----------------------------------------------------------------------
                # *** Continue stimulus presentation

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

                if stimulus_jitter > 0.0:
                    # Randomly sample stimulus duration jitter for the current trial.
                    stimulus_jitter_current_trial = np.random.uniform(
                        low=0.0,
                        high=stimulus_jitter,
                    )
                else:
                    stimulus_jitter_current_trial = 0.0

                # Wait for stimulus duration.
                t_stim_end_expected = (
                    t_stim_start  # Stimulus start time
                    + stimulus_duration  # Regular stimulus duration
                    + extra_stimulus_duration  # Extra stimulus duration for long words
                    + stimulus_jitter_current_trial  # Random stimulus duration jitter
                )

                # The data from the current stimulus will be logged *after* flipping the
                # screen for the next stimulus. Keep a deepcopy so as to log the
                # parameters of the current stimulus (not the next one).
                stimulus_data = deepcopy(
                    {
                        "stimulus_start_time": t_stim_start,
                        "word": word,
                        "font_name": font_name,
                        "font_size": font_size,
                        "font_is_bold": font_is_bold,
                        "font_is_italic": font_is_italic,
                        "font_spacing": font_spacing,
                        "font_color": font_color,
                    }
                )
                need_to_log_previous_stimulus = True

                stimulus_block_counter += 1

                # Continue stimulus presentation until the current stimulus time is up.
                while eeg_device.lsl_local_clock() < t_stim_end_expected:
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

                # Log EEG data.
                eeg_data, eeg_ts = eeg_device.get_board_data()
                if eeg_data.size > 0:
                    data_logging_queue.put(
                        {
                            "type": "eeg",
                            "eeg_data": eeg_data,
                            "eeg_timestamps": eeg_ts,
                        }
                    )

                # ----------------------------------------------------------------------
                # *** Inter-block interval

                if stimulus_block_counter == stimuli_per_block:
                    # Inter-block interval (break).
                    screen.fill(text_config.rest_condition_color)
                    pygame.display.flip()
                    # Start of inter-block interval.
                    t_ibi_start = eeg_device.lsl_local_clock()

                    if use_ibi_audio_cue:
                        # Audio cue to signal the beginning of the inter-block interval.
                        pure_tone_start.play()

                    # ------------------------------------------------------------------
                    # *** Log previous stimulus

                    # The end time of the previous stimulus is the onset of the current
                    # inter-block interval.
                    t_stim_end_actual = t_ibi_start

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

                    stimulus_data["stimulus_end_time"] = t_stim_end_actual
                    stimulus_data["stimulus_duration_s"] = (
                        t_stim_end_actual - stimulus_data["stimulus_start_time"]
                    )

                    data_logging_queue.put(
                        {"type": "stimulus", "stimulus_data": stimulus_data}
                    )

                    need_to_log_previous_stimulus = False

                    # ------------------------------------------------------------------
                    # *** Continue inter-block interval

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

                    continue

                # ----------------------------------------------------------------------
                # *** ISI

                # Compute the duration of the upcoming inter stimulus interval (ISI).
                # ISI duration can be zero.
                next_isi_duration = isi_duration
                if isi_jitter > 0.0:
                    # Randomly sample ISI duration jitter for the current trial.
                    isi_jitter_current_trial = np.random.uniform(
                        low=0.0,
                        high=isi_jitter,
                    )
                    next_isi_duration += isi_jitter_current_trial

                # The ISI interval can be zero; in that case, do not include an ISI at
                # all.
                if next_isi_duration < 0.0167:
                    # Skip ISI if ISI duration is less than one frame (assuming 60 Hz
                    # refresh rate). The stimulus stays on screen for now.
                    print("Skipping ISI")
                    t_stim_end_actual = None  # No ISI, the stimulus is still shown
                    continue

                # End of stimulus presentation. Display ISI grey screen.
                screen.fill(text_config.rest_condition_color)
                pygame.display.flip()
                t_stim_end_actual = eeg_device.lsl_local_clock()
                # Time until when to show grey screen (ISI).
                t_isi_end = t_stim_end_actual + next_isi_duration

                # Continue checking for quit events.
                while eeg_device.lsl_local_clock() < t_isi_end:
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

                # Send post-stimulus EEG data (to avoid buffer overflow).
                eeg_data, eeg_ts = eeg_device.get_board_data()
                if eeg_data.size > 0:
                    data_logging_queue.put(
                        {"type": "eeg", "eeg_data": eeg_data, "eeg_timestamps": eeg_ts}
                    )

            # --------------------------------------------------------------------------
            # *** Log final stimulus data

            if need_to_log_previous_stimulus:
                if t_stim_end_actual is None:
                    screen.fill(text_config.rest_condition_color)
                    pygame.display.flip()
                    t_stim_end_actual = eeg_device.lsl_local_clock()

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

                stimulus_data["stimulus_end_time"] = t_stim_end_actual
                stimulus_data["stimulus_duration_s"] = (
                    t_stim_end_actual - stimulus_data["stimulus_start_time"]
                )

                data_logging_queue.put(
                    {"type": "stimulus", "stimulus_data": stimulus_data}
                )

            # --------------------------------------------------------------------------
            # *** Show multiple choice questions

            n_correct_answers = 0
            n_questions = len(questions_and_answers)

            qa_font = pygame.font.SysFont("arial", 36)
            feedback_font = pygame.font.SysFont("arial", 60, bold=True)

            # Loop through questions.
            for q_idx, q_data in enumerate(questions_and_answers):
                if not running:
                    break

                question_text = q_data["question"]
                answers = q_data["answers"]

                answered = False

                while not answered and running:
                    # Clear screen for the question.
                    screen.fill(text_config.rest_condition_color)

                    y_pos = int(screen_height * 0.15)

                    y_pos = draw_text_wrapped(
                        surface=screen,
                        text=question_text,
                        font=qa_font,
                        color=(255, 255, 255),
                        y_start=y_pos,
                        max_width=screen_width * 0.8,
                        screen_width=screen_width,
                    )
                    y_pos += 60  # Add extra spacing before options

                    # Draw answer options.
                    for a_idx, ans_data in enumerate(answers):
                        ans_text = f"[{a_idx + 1}] {ans_data['answer']}"
                        y_pos = draw_text_wrapped(
                            surface=screen,
                            text=ans_text,
                            font=qa_font,
                            color=(255, 255, 255),
                            y_start=y_pos,
                            max_width=screen_width * 0.8,
                            screen_width=screen_width,
                        )
                        y_pos += 30  # Spacing between answers

                    pygame.display.flip()

                    # Wait for participant input.
                    for event in pygame.event.get():
                        if event.type == pygame.QUIT:
                            running = False
                        elif event.type == pygame.KEYDOWN:
                            if event.key == pygame.K_ESCAPE:
                                running = False
                            # Map keys 1-4 (standard number row or numpad) to answer
                            # indices 0-3.
                            elif (
                                pygame.K_1 <= event.key <= pygame.K_9
                                or pygame.K_KP1 <= event.key <= pygame.K_KP9
                            ):
                                # Determine which number was pressed, handling both top
                                # row and numpad.
                                if pygame.K_1 <= event.key <= pygame.K_9:
                                    selected_idx = event.key - pygame.K_1
                                else:
                                    selected_idx = event.key - pygame.K_KP1

                                print(
                                    f"event.key: {event.key} | selected_idx: {selected_idx}"
                                )

                                # Check if the pressed key corresponds to a valid
                                # option.
                                if selected_idx < len(answers):
                                    is_correct = answers[selected_idx]["correct"]

                                    if is_correct:
                                        n_correct_answers += 1
                                        feedback_text = "Correct"
                                        feedback_color = (0, 255, 0)  # Green
                                    else:
                                        feedback_text = "Incorrect"
                                        feedback_color = (255, 0, 0)  # Red

                                    answered = True

                # Display feedback (whether the answer was correct).
                if running:
                    screen.fill(text_config.rest_condition_color)
                    feedback_surface = feedback_font.render(
                        feedback_text, True, feedback_color
                    )
                    feedback_rect = feedback_surface.get_rect(
                        center=(screen_width // 2, screen_height // 2)
                    )
                    screen.blit(feedback_surface, feedback_rect)
                    pygame.display.flip()

                    # Pause for participant to read the feedback.
                    pygame.time.delay(2000)

            # Write behavioural results to hdf5 file.
            behavioural_data = {
                "n_questions": n_questions,
                "n_answers": n_answers,  # Answer options per question
                "n_correct_answers": n_correct_answers,
            }
            data_logging_queue.put(
                {"type": "behavioural", "behavioural_data": behavioural_data}
            )

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
