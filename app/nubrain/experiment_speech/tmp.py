import time
import traceback

import pygame

pygame.mixer.init()

audio_filepath = "/home/john/Dropbox/Start_Ups/nubrain/data/stimuli/audio/Doyle_1905_The_Return_of_Sherlock_Holmes_v3/section_006_Eric_Fast.ogg"
sample_rate = 24000

rest_condition_color = (0, 0, 0)

running = True
while running:
    pygame.init()

    pygame.mixer.init(frequency=sample_rate, size=-16, channels=2, buffer=256)

    # Get screen dimensions and set up full screen.
    screen_info = pygame.display.Info()
    screen_width = screen_info.current_w
    screen_height = screen_info.current_h
    screen = pygame.display.set_mode((screen_width, screen_height), pygame.FULLSCREEN)
    pygame.display.set_caption("Audio presentation")
    pygame.mouse.set_visible(False)

    try:
        # Initial grey screen.
        pygame.time.wait(100)
        screen.fill(rest_condition_color)
        pygame.display.flip()
        pygame.time.wait(100)

        sound = pygame.mixer.Sound(audio_filepath)
        start_time = time.time()
        channel = sound.play()

        # Wait until the sound finishes playing.
        while channel.get_busy():
            pygame.event.pump()
            pygame.time.wait(1)  # avoid busy-waiting

        end_time = time.time()
        pygame.mixer.quit()

        running = False

    except Exception as e:
        print(f"An error occurred: {e}")
        print(traceback.format_exc())
        running = False
    finally:
        pygame.quit()

audio_duration = round(end_time - start_time, 3)
print(f"Audio presentation duration: {audio_duration} seconds")
