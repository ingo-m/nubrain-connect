"""
Demo mode for instructing participants. Does not use EEG device, only presents a short
series of text stimuli (i.e. words).
"""

import json
import random
import traceback
from copy import deepcopy
from time import time

import numpy as np
import pygame

from nubrain.audio.tone import generate_tone
from nubrain.experiment_text.text_config import TextConfig
from nubrain.text.rendering import construct_fonts, render_spaced_text


def text_demo(config: dict):
    # ----------------------------------------------------------------------------------
    # *** Get config

    subject_id = config["subject_id"]
    session_id = config["session_id"]

    path_stimuli = config["path_stimuli"]

    initial_rest_duration = config["initial_rest_duration"]
    stimulus_duration = config["stimulus_duration"]
    stimulus_jitter = config["stimulus_jitter"]
    stimulus_extension_target = config["stimulus_extension_target"]
    isi_duration = config["isi_duration"]
    isi_jitter = config["isi_jitter"]
    isi_extension_target = config["isi_extension_target"]
    inter_block_rest_duration = config["inter_block_rest_duration"]
    n_chars_long_word_threshold = config["n_chars_long_word_threshold"]
    extra_duration_per_char = config["extra_duration_per_char"]
    max_extra_stimulus_duration = config["max_extra_stimulus_duration"]

    section_idx_start = config["section_idx_start"]
    n_sections_to_show = config["n_sections_to_show"]

    stimuli_per_block = config["stimuli_per_block"]
    stimulus_font_sizes = config["stimulus_font_sizes"]
    stimulus_font_min_spacing = config["stimulus_font_min_spacing"]
    stimulus_font_max_spacing = config["stimulus_font_max_spacing"]

    text_config = TextConfig()

    # ----------------------------------------------------------------------------------
    # *** Load stimulus data from JSON file

    with open(path_stimuli, "r", encoding="utf-8") as file:
        stimuli = json.load(file)

    text_sections = stimuli["text_sections"]

    # Select subset of text.
    text_sections = text_sections[
        section_idx_start : (section_idx_start + n_sections_to_show)
    ]

    text = [x for xs in [x["text_with_targets"] for x in text_sections] for x in xs]
    is_target = [x for xs in [x["is_target"] for x in text_sections] for x in xs]

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

            # Pause for specified number of milliseconds.
            pygame.time.delay(int(round(initial_rest_duration * 1000.0)))

            # Count stimuli to introduce breaks between blocks.
            stimulus_block_counter = 0
            t_stim_end_actual = None
            need_to_log_previous_stimulus = False

            # Loop through words.
            for word, is_target_event in zip(text, is_target):
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
                t_stim_start = time()

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

                    dummy_log = {
                        "type": "marker",
                        "marker_value": text_config.stim_end_marker,
                        "timestamp": t_stim_end_actual,
                    }
                    print(f"data_logging_queue.put: {dummy_log}")

                    stimulus_data["response_time_s"] = response_time
                    stimulus_data["stimulus_end_time"] = t_stim_end_actual
                    stimulus_data["stimulus_duration_s"] = (
                        t_stim_end_actual - stimulus_data["stimulus_start_time"]
                    )

                    dummy_log = {"type": "stimulus", "stimulus_data": stimulus_data}
                    print(f"data_logging_queue.put: {dummy_log}")

                # ----------------------------------------------------------------------
                # *** Continue stimulus presentation

                dummy_log = {
                    "type": "marker",
                    "marker_value": text_config.stim_start_marker,
                    "timestamp": t_stim_start,
                }
                print(f"data_logging_queue.put: {dummy_log}")

                if stimulus_jitter > 0.0:
                    # Randomly sample stimulus duration jitter for the current trial.
                    stimulus_jitter_current_trial = np.random.uniform(
                        low=0.0,
                        high=stimulus_jitter,
                    )
                else:
                    stimulus_jitter_current_trial = 0.0

                response_made = False
                response_time = np.nan
                response_deadline = (
                    t_stim_start  # Stimulus start time
                    + stimulus_duration  # Regular stimulus duration
                    + extra_stimulus_duration  # Extra stimulus duration for long words
                    + stimulus_extension_target  # Extra stimulus duration for targets
                    + stimulus_jitter_current_trial  # Random stimulus duration jitter
                    + isi_duration  # Inter stimulus interval (can be zero)
                    + isi_extension_target  # Extra ISI for targets (can be zero)
                )

                # Wait for stimulus duration, but check for responses continuously.
                t_stim_end_expected = (
                    t_stim_start  # Stimulus start time
                    + stimulus_duration  # Regular stimulus duration
                    + extra_stimulus_duration  # Extra stimulus duration for long words
                    + stimulus_jitter_current_trial  # Random stimulus duration jitter
                )
                if is_target_event:
                    # Extra stimulus duration for targets.
                    t_stim_end_expected += stimulus_extension_target

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
                        "is_target_event": is_target_event,
                    }
                )
                need_to_log_previous_stimulus = True

                stimulus_block_counter += 1

                # Continue stimulus presentation until the current stimulus time is up.
                while time() < t_stim_end_expected:
                    for event in pygame.event.get():
                        if event.type == pygame.QUIT:
                            running = False
                        if event.type == pygame.KEYDOWN:
                            keydown_time = time()
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

                # ----------------------------------------------------------------------
                # *** Inter-block interval

                if stimulus_block_counter == stimuli_per_block:
                    # Inter-block interval (break).
                    screen.fill(text_config.rest_condition_color)
                    pygame.display.flip()
                    # Start of inter-block interval.
                    t_ibi_start = time()

                    if use_ibi_audio_cue:
                        # Audio cue to signal the beginning of the inter-block interval.
                        pure_tone_start.play()

                    # ------------------------------------------------------------------
                    # *** Log previous stimulus

                    # The end time of the previous stimulus is the onset of the current
                    # inter-block interval.
                    t_stim_end_actual = t_ibi_start

                    dummy_log = {
                        "type": "marker",
                        "marker_value": text_config.stim_end_marker,
                        "timestamp": t_stim_end_actual,
                    }
                    print(f"data_logging_queue.put: {dummy_log}")

                    stimulus_data["response_time_s"] = response_time
                    stimulus_data["stimulus_end_time"] = t_stim_end_actual
                    stimulus_data["stimulus_duration_s"] = (
                        t_stim_end_actual - stimulus_data["stimulus_start_time"]
                    )

                    dummy_log = {"type": "stimulus", "stimulus_data": stimulus_data}
                    print(f"data_logging_queue.put: {dummy_log}")

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

                    while time() < t_ibi_end:  # Continue inter-block interval?
                        if (
                            use_ibi_audio_cue
                            and (t_ibi_end_audio_cue <= time())
                            and not ibi_end_cue_played_yet
                        ):  # Time to play end of inter-block interval cue?
                            # Play the cue to signal the end of the inter-block
                            # interval.
                            pure_tone_end.play()
                            ibi_end_cue_played_yet = True

                    stimulus_block_counter = 0

                    continue

                # ----------------------------------------------------------------------
                # *** ISI

                # Compute the duration of the upcoming inter stimulus interval (ISI).
                # ISI duration can be zero.
                next_isi_duration = isi_duration
                if is_target_event:
                    # If this is a target event, prolong the ISI duration, to allow the
                    # subject to respond before the onset of the next stimulus, to
                    # reduce the probability of a motor response artefact in the
                    # subsequent trial.
                    next_isi_duration += isi_extension_target
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
                t_stim_end_actual = time()
                # Time until when to show grey screen (ISI).
                t_isi_end = t_stim_end_actual + next_isi_duration

                # Continue checking for late responses or quit events.
                while time() < t_isi_end:
                    for event in pygame.event.get():
                        if event.type == pygame.QUIT:
                            running = False
                        if event.type == pygame.KEYDOWN:
                            keydown_time = time()
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

            # --------------------------------------------------------------------------
            # *** Log final stimulus data

            if need_to_log_previous_stimulus:
                if t_stim_end_actual is None:
                    screen.fill(text_config.rest_condition_color)
                    pygame.display.flip()
                    t_stim_end_actual = time()

                dummy_log = {
                    "type": "marker",
                    "marker_value": text_config.stim_end_marker,
                    "timestamp": t_stim_end_actual,
                }
                print(f"data_logging_queue.put: {dummy_log}")

                stimulus_data["response_time_s"] = response_time
                stimulus_data["stimulus_end_time"] = t_stim_end_actual
                stimulus_data["stimulus_duration_s"] = (
                    t_stim_end_actual - stimulus_data["stimulus_start_time"]
                )

                dummy_log = {"type": "stimulus", "stimulus_data": stimulus_data}
                print(f"data_logging_queue.put: {dummy_log}")

            # --------------------------------------------------------------------------
            # *** Show behavioural results

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
            dummy_log = {"type": "behavioural", "behavioural_data": behavioural_data}
            print(f"data_logging_queue.put: {dummy_log}")

            if running:
                # Display behavioural results.
                screen.fill(text_config.rest_condition_color)

                # Behavioural results title.
                title_font = pygame.font.Font(None, 72)
                title_text = title_font.render(
                    "Experiment Complete",
                    True,
                    text_config.font_colors[0],
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
                    text_config.font_colors[0],
                )
                misses_text = results_font.render(
                    f"Misses: {n_misses}",
                    True,
                    text_config.font_colors[0],
                )
                false_alarms_text = results_font.render(
                    f"False Alarms: {n_false_alarms}",
                    True,
                    text_config.font_colors[0],
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

        except Exception as e:
            print(f"An error occurred during the experiment: {e}")
            print(traceback.format_exc())
            running = False
        finally:
            pygame.quit()
            print("Experiment closed.")
