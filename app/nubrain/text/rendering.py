import pygame


def construct_fonts(
    *,
    font_sizes: list[int],
) -> list[dict]:
    """
    Generate a list of pygame fonts with different font types and settings (normal,
    bold, italic). Used to render text with different appearance in terms of low-level
    visual features.

    Note: You need to run `pygame.init()` in the parent function before constructing the
    fonts.
    """
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

    fonts = []

    for font_name in font_names:
        for font_size in font_sizes:
            # Some bold fonts are not rendered nicely, let's skip bold.
            # for is_bold in [True, False]:
            is_bold = False
            for is_italic in [True, False]:
                new_font = pygame.font.SysFont(
                    font_name,
                    font_size,
                    is_bold,
                    is_italic,
                )
                fonts.append(
                    {
                        "font_name": font_name,
                        "font_size": font_size,
                        "font_is_bold": is_bold,
                        "font_is_italic": is_italic,
                        "font": new_font,
                    }
                )

    return fonts


def render_spaced_text(
    *,
    text: str,
    font: pygame.font.SysFont,
    color: tuple,
    spacing: float,
) -> pygame.Surface:
    """
    Renders text with custom letter spacing.

    Checks the width of every specific character as it is rendered. This is necessary
    because the width of characters differs between characters and fonts. Returns a
    pygame surface containing the spaced text.
    """
    if not text:
        return pygame.Surface((0, 0))

    # Render each character to its own surface.
    char_surfaces = [font.render(char, True, color) for char in text]

    # Calculate the total width and maximum height needed. Total width = sum of all char
    # widths + spacing between each char.
    total_width = sum([surf.get_width() for surf in char_surfaces]) + (
        spacing * (len(text) - 1)
    )
    max_height = max([surf.get_height() for surf in char_surfaces])

    # Create a master transparent surface to hold the word. SRCALPHA ensures the
    # background of this new surface remains transparent.
    word_surface = pygame.Surface((total_width, max_height), pygame.SRCALPHA)

    # Blit each character onto the master surface with the appropriate spacing offset.
    current_x = 0
    for surf in char_surfaces:
        word_surface.blit(surf, (current_x, 0))
        current_x += surf.get_width() + spacing

    return word_surface
