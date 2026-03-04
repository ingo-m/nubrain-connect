import sys

import pygame

# 1. Initialize Pygame
pygame.init()

# 2. Configuration & Setup
WIDTH, HEIGHT = 1600, 600
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Word Presentation")

BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
DURATION_MS = 2000

# text = "The rain had finally stopped and dawn light was seeping through the wet cloth over his eyes when Catelyn Stark gave the command to dismount."
# words = text.split(" ")

font_names = [
    "andalemono",
    "arial",
    "arialblack",
    "c059",
    "comicsansms",
    "couriernew",
    "dejavuserif",
    "georgia",
    "impact",
    "liberationmono",
    "liberationsans",
    "liberationsansnarrow",
    "liberationserif",
    "nimbusmonops",
    "nimbusroman",
    "nimbussans",
    "nimbussansnarrow",
    "notomono",
    "notosans",
    "notosanscjkhk",
    "notosansdisplay",
    "notosansmono",
    "notosansmonocjktc",
    "notoserif",
    "notoserifcjksc",
    "notoserifdisplay",
    "p052",
    "timesnewroman",
    "trebuchetms",
    "tuffy",
    "ubuntu",
    "ubuntumono",
    "ubuntusansmono",
    "urwbookman",
    "urwgothic",
    "verdana",
    "z003",
]

# 3. Setup the list of diverse fonts
# We define combinations of (font_family, is_bold, is_italic)
# font_configurations = [
#     ("arial", False, False),  # Normal
#     ("arial", True, False),  # Bold
#     ("arial", False, True),  # Italic
#     ("timesnewroman", False, False),
#     ("timesnewroman", True, False),
#     ("timesnewroman", False, True),
#     ("courier", False, False),
#     ("courier", True, False),
#     ("courier", False, True),
# ]

words = font_names

# Create actual Pygame font objects from the configurations
fonts = []
font_size = 48

is_bold = False
is_italic = False
# TODO: is_wide

# for name, is_bold, is_italic in font_configurations:
# for font_name in font_names:
#     try:
#         font = pygame.font.SysFont(font_name, font_size, is_bold, is_italic)
#         fonts.append(font)
#     except Exception as e:
#         print(f"Warning: Could not load font {font_name}. Error: {e}")

# 4. Main Loop Setup
word_index = 0
running = True

# Set the initial timer in the past so the first word draws immediately
last_update = pygame.time.get_ticks() - DURATION_MS

while running:
    # Handle window events (like clicking the 'X' to close)
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    # Check if we have exhausted the word list
    if word_index >= len(words):
        pygame.time.wait(1000)  # Pause for a second at the very end
        running = False
        continue

    # 5. Timing and Rendering Logic
    current_time = pygame.time.get_ticks()

    if current_time - last_update >= DURATION_MS:
        # Clear the screen (Black background)
        screen.fill(BLACK)

        # Get the next word and randomly select a font
        current_word = words[word_index] + "123 ,-.:;“”'‘’()!?…" + '"'
        # selected_font = random.choice(fonts)
        print(font_names[word_index])
        selected_font = pygame.font.SysFont(
            font_names[word_index], font_size, is_bold, is_italic
        )

        # Render the text (White font)
        text_surface = selected_font.render(current_word, True, WHITE)

        # Center the text on the screen
        text_rect = text_surface.get_rect(center=(WIDTH // 2, HEIGHT // 2))

        # Blit (draw) the text surface onto the main screen
        screen.blit(text_surface, text_rect)
        pygame.display.flip()  # Update the actual display

        # Increment index and reset the timer
        word_index += 1
        last_update = current_time

# Clean exit
pygame.quit()
sys.exit()
