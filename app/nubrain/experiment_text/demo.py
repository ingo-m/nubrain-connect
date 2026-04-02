"""
Demo mode for instructing participants. Does not use EEG device, only presents a short
series of text stimuli (i.e. words).
"""

import random
import traceback
from time import time

import numpy as np
import pygame

from nubrain.audio.tone import generate_tone
from nubrain.experiment_text.random_target_events import sample_target_events
from nubrain.experiment_text.text_config import TextConfig
from nubrain.text.rendering import construct_fonts, render_spaced_text
from nubrain.text.tools import load_and_preprocess_text


def text_demo(config: dict):
    # ----------------------------------------------------------------------------------
    # *** Get config

    device_type = config["device_type"]

    path_text = config["path_text"]

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
    stimulus_font_sizes = config["stimulus_font_sizes"]
    stimulus_font_min_spacing = config["stimulus_font_min_spacing"]
    stimulus_font_max_spacing = config["stimulus_font_max_spacing"]

    text_config = TextConfig()

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

            stimulus_fonts = construct_fonts(font_sizes=stimulus_font_sizes)

            # Pause for specified number of milliseconds.
            pygame.time.delay(int(round(initial_rest_duration * 1000.0)))

            # Count stimuli to introduce breaks between blocks.
            stimulus_block_counter = 0

            # Loop through words.
            for word, is_target_event in zip(text, is_target):
                if not running:  # Check for quit event
                    break

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
                pygame.display.flip()
                # Start of stimulus presentation.
                t_stim_start = time()

                response_made = False
                response_time = np.nan
                response_deadline = t_stim_start + response_window_duration

                # Wait for stimulus duration, but check for responses continuously.
                t_stim_end_expected = t_stim_start + stimulus_duration
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

                # End of stimulus presentation. Display ISI grey screen.
                screen.fill(text_config.rest_condition_color)
                pygame.display.flip()
                t_stim_end_actual = time()

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

                if not running:
                    break

                stimulus_block_counter += 1

                if stimulus_block_counter == stimuli_per_block:
                    # Inter-block grey screen.
                    screen.fill(text_config.rest_condition_color)
                    pygame.display.flip()
                    # Start of inter-block interval.
                    t_ibi_start = time()

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

            # End of word loop.

            # Calculate behavioural results.
            n_misses = n_total_targets - n_hits

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
