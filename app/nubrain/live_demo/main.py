"""
Live demo with cached data.
"""

import io
import pickle
import traceback
from time import time

import numpy as np
import pygame

from nubrain.experiment.global_config import GlobalConfig
from nubrain.experiment_eeg_to_image_v1.tone import generate_tone
from nubrain.image.tools import (
    scale_image_surface,
)


def run_live_demo(cache: str):
    # ----------------------------------------------------------------------------------
    # *** Get config

    initial_rest_duration = 0.2
    pre_stimulus_interval = 1.1
    # image_duration = 1.1
    post_stimulus_interval = 0.7
    # generated_image_duration = 3.0
    isi_jitter = 0.1
    inter_block_grey_duration = 0.1

    image_generation_step_delay = 0.2

    global_config = GlobalConfig()

    # ----------------------------------------------------------------------------------
    # *** Load cache

    with open(cache, "rb") as f:
        trial_data = pickle.load(f)

    n_trials = len(trial_data)

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

        # Create a sound object from the numpy array.
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

        # Prepare text.
        font = pygame.font.Font(None, 56)

        text_original = font.render("Original image", True, (0, 0, 0))
        text_reconstructed = font.render("Reconstructed from EEG", True, (0, 0, 0))

        text_original_rect = text_original.get_rect(
            center=((screen_width * 1 // 4), (screen_height // 4 - 50))
        )
        text_reconstructed_rect = text_reconstructed.get_rect(
            center=((screen_width * 3 // 4), (screen_height // 4 - 50))
        )

        screen.blit(text_original, text_original_rect)
        screen.blit(text_reconstructed, text_reconstructed_rect)

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

            for idx_trial in range(n_trials):
                # Original stimulus image (as bytes).
                stimulus_image_bytes = trial_data[idx_trial]["stimulus_image_bytes"]
                # List of reconstructed image steps (as bytes).
                generated_images_bytes = trial_data[idx_trial]["generated_images_bytes"]

                stimulus_image_file = io.BytesIO(stimulus_image_bytes)

                current_image = pygame.image.load(stimulus_image_file)

                current_image = scale_image_surface(
                    image_surface=current_image,  # TODO
                    screen_width=screen_width,
                    screen_height=screen_height,
                )

                # Play tone to cue block start.
                pure_tone.play()
                pygame.time.delay(int(round(tone_pre_stimulus_onset * 1000.0)))

                # ----------------------------------------------------------------------
                # *** Pre-stimulus interval

                if not running:  # Check for quit event
                    break

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

                # ----------------------------------------------------------------------
                # *** Stimulus period

                pygame.display.flip()
                # t_stim_start = time()  # Start of stimulus presentation.

                # Wait for user input (space bar press) to continue.
                waiting = True
                while waiting:
                    # Process all events in the queue.
                    for event in pygame.event.get():
                        if event.type == pygame.QUIT:
                            running = False
                            waiting = False
                        if event.type == pygame.KEYDOWN:
                            if event.key == pygame.K_ESCAPE:
                                running = False
                                waiting = False
                            elif event.key == pygame.K_SPACE:
                                # Exit the loop	after space bar press.
                                waiting = False
                    if not running:
                        break
                    pygame.time.wait(100)
                if not running:
                    break

                # ----------------------------------------------------------------------
                # *** Post-stimulus period

                # End of stimulus presentation. Display grey screen.
                screen.fill(global_config.rest_condition_color)
                pygame.display.flip()
                t_stim_end_actual = time()

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

                # ----------------------------------------------------------------------
                # *** Show generated images next to the original image

                for idx_step in range(len(generated_images_bytes)):
                    print(f"Step {idx_step}")
                    generated_image_bytes = generated_images_bytes[idx_step]
                    generated_image_file = io.BytesIO(generated_image_bytes)

                    # Create a Pygame surface
                    generated_image_surface = pygame.image.load(
                        generated_image_file
                    )  # .convert()

                    # Scale the image for display.
                    generated_image_surface = scale_image_surface(
                        image_surface=generated_image_surface,
                        screen_width=screen_width,
                        screen_height=screen_height,
                    )

                    # Display the original image on the left, and the generated
                    # image on the right.
                    original_img_rect = current_image.get_rect(
                        center=(
                            (screen_width * 1 // 4),
                            screen_height // 2 + 50,
                        )
                    )
                    generated_img_rect = generated_image_surface.get_rect(
                        center=(
                            (screen_width * 3 // 4),
                            screen_height // 2 + 50,
                        )
                    )

                    screen.fill(global_config.rest_condition_color)
                    # Text titles (original & reconstructed image).
                    screen.blit(text_original, text_original_rect)
                    screen.blit(text_reconstructed, text_reconstructed_rect)
                    # Original image.
                    screen.blit(current_image, original_img_rect)
                    # Reconstructed image.
                    screen.blit(generated_image_surface, generated_img_rect)

                    pygame.display.flip()

                    pygame.time.delay(int(round(image_generation_step_delay * 1000.0)))

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
                # t_generated_img_end = time() + generated_image_duration

                screen.fill(global_config.rest_condition_color)
                # Text titles (original & reconstructed image).
                screen.blit(text_original, text_original_rect)
                screen.blit(text_reconstructed, text_reconstructed_rect)
                # Original image.
                screen.blit(current_image, original_img_rect)
                # Reconstructed image.
                screen.blit(generated_image_surface, generated_img_rect)

                pygame.display.flip()

                # Show generated image for specified amount of time.
                # while time() < t_generated_img_end:
                #     for event in pygame.event.get():
                #         if event.type == pygame.QUIT:
                #             running = False
                #         if event.type == pygame.KEYDOWN:
                #             if event.key == pygame.K_ESCAPE:
                #                 running = False

                # Wait for user input (space bar press) to continue.
                waiting = True
                while waiting:
                    # Process all events in the queue.
                    for event in pygame.event.get():
                        if event.type == pygame.QUIT:
                            running = False
                            waiting = False
                        if event.type == pygame.KEYDOWN:
                            if event.key == pygame.K_ESCAPE:
                                running = False
                                waiting = False
                            elif event.key == pygame.K_SPACE:
                                # Exit the loop	after space bar press.
                                waiting = False
                    if not running:
                        break
                    pygame.time.wait(100)
                if not running:
                    break

                # ----------------------------------------------------------------------
                # *** Inter-trial grey screen

                # End of generated image presentation. Display grey screen.
                screen.fill(global_config.rest_condition_color)
                pygame.display.flip()
                remaining_wait = (
                    inter_block_grey_duration
                    - tone_pre_stimulus_onset
                    + np.random.uniform(low=0.0, high=isi_jitter)
                )
                pygame.time.delay(int(round(remaining_wait * 1000.0)))

            # --------------------------------------------------------------------------
            # *** End of experiment

            running = False

        except Exception as e:
            print(f"An error occurred during the experiment: {e}")
            print(traceback.format_exc())
            running = False
        finally:
            pygame.quit()
            print("Experiment closed.")


# run_live_demo(
#     cache="/media/john/data_drive/nubrain/inference/e49b81039572c4031bb6016b1c75f71b896345e6_live_demo/cached.pickle"
# )
