import glob
import json
import os

import pygame


def main(*, data_directory: str):
    # ----------------------------------------------------------------------------------
    # *** Find input files

    # Find all .ogg files first
    search_pattern_ogg = os.path.join(data_directory, "*.ogg")
    ogg_files = sorted(glob.glob(search_pattern_ogg))

    if not ogg_files:
        raise AssertionError(f"No .ogg files found in {data_directory}")

    # Assert that every .ogg file has a corresponding _timestamps.json file.
    for ogg_path in ogg_files:
        base_name = ogg_path.replace(".ogg", "")
        expected_json_path = f"{base_name}_timestamps.json"

        if not os.path.exists(expected_json_path):
            raise AssertionError(
                f"Missing timestamp file for audio: {os.path.basename(ogg_path)}. "
                f"Expected to find {os.path.basename(expected_json_path)}"
            )

    # ----------------------------------------------------------------------------------
    # *** Pygame setup

    pygame.init()
    pygame.mixer.init()

    screen_width, screen_height = 800, 400
    screen = pygame.display.set_mode((screen_width, screen_height))
    pygame.display.set_caption("Audio-Text Sync Player")

    font = pygame.font.SysFont(None, 72)

    # ----------------------------------------------------------------------------------
    # *** Presentation

    for ogg_path in ogg_files:
        base_name = ogg_path.replace(".ogg", "")
        json_path = f"{base_name}_timestamps.json"

        # Load the timestamps.
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
            words_data = data.get("aligned_words", [])

        if not words_data:
            print(f"WARNING: No words found in {os.path.basename(json_path)}")
            continue

        print(f"Playing {os.path.basename(ogg_path)}")

        # Load and play the audio.
        pygame.mixer.music.load(ogg_path)
        pygame.mixer.music.play()

        current_word_idx = 0
        total_words = len(words_data)
        running = True
        skip_file = False

        # Loop while audio is playing.
        while running and pygame.mixer.music.get_busy():
            # Handle Pygame events.
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    return None
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:  # Press ESC to skip
                        skip_file = True
                        running = False

            if skip_file:
                pygame.mixer.music.stop()
                break

            # Get current audio position in seconds.
            current_time_sec = pygame.mixer.music.get_pos() / 1000.0

            current_word = ""

            # Fast-forward index if audio is past current word's end time.
            while (
                current_word_idx < total_words
                and current_time_sec > words_data[current_word_idx]["end"]
            ):
                current_word_idx += 1

            # Check if we are inside the active window for the current word.
            if current_word_idx < total_words:
                word_info = words_data[current_word_idx]
                if current_time_sec >= word_info["start"]:
                    current_word = word_info["word"].strip()

            screen.fill((30, 30, 30))  # Dark gray background

            if current_word:
                text_surface = font.render(current_word, True, (255, 255, 255))
                text_rect = text_surface.get_rect(
                    center=(screen_width // 2, screen_height // 2)
                )
                screen.blit(text_surface, text_rect)

            pygame.display.flip()
            pygame.time.Clock().tick(60)

        # Brief pause between files.
        pygame.time.delay(500)

    print("Finished playing all files.")
    pygame.quit()


if __name__ == "__main__":
    data_directory = "/home/john/Dropbox/Deep_Learning/TTS/Qwen3-TTS-12Hz-1.7B-CustomVoice/transcript_test"

    main(data_directory=data_directory)
