import time
import traceback

import pygame

pygame.mixer.init()

audio_files = [
    "/home/john/Dropbox/Deep_Learning/TTS/Qwen3-TTS-12Hz-1.7B-CustomVoice/daenerys_Sohee_None.ogg",
    "/home/john/Dropbox/Deep_Learning/TTS/Qwen3-TTS-12Hz-1.7B-CustomVoice/jon_Sohee_None.ogg",
    "/home/john/Dropbox/Deep_Learning/TTS/Qwen3-TTS-12Hz-1.7B-CustomVoice/nine_days_Sohee_None.ogg",
]
timestamps = []

rest_condition_color = [0, 0, 0]


running = True
while running:
    pygame.init()

    # Get screen dimensions and set up full screen.
    screen_info = pygame.display.Info()
    screen_width = screen_info.current_w
    screen_height = screen_info.current_h
    screen = pygame.display.set_mode((screen_width, screen_height), pygame.FULLSCREEN)
    pygame.display.set_caption("Silent Reading Experiment")
    pygame.mouse.set_visible(False)

    try:
        # Initial grey screen.
        pygame.time.wait(100)
        screen.fill(rest_condition_color)
        pygame.display.flip()
        pygame.time.wait(100)

        for filepath in audio_files:
            sound = pygame.mixer.Sound(filepath)
            start_time = time.time()
            channel = sound.play()

            # Wait until this sound finishes playing.
            while channel.get_busy():
                pygame.time.wait(10)  # avoid busy-waiting
            end_time = time.time()

            timestamps.append(
                {
                    "file": filepath,
                    "start": start_time,
                    "end": end_time,
                }
            )

        pygame.mixer.quit()

        for entry in timestamps:
            print(
                f"{entry['file']}: start={entry['start']:.3f}, end={entry['end']:.3f}"
            )

        running = False

    except Exception as e:
        print(f"An error occurred during the experiment: {e}")
        print(traceback.format_exc())
        running = False
    finally:
        pygame.quit()
        print("Experiment closed.")
