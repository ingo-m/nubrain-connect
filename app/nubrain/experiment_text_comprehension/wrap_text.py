def draw_text_wrapped(*, surface, text, font, color, y_start, max_width, screen_width):
    """
    Helper function to handle text wrapping in pygame.
    """
    words = text.split(" ")
    lines = []
    current_line = []

    for word in words:
        test_line = " ".join(current_line + [word])
        if font.size(test_line)[0] <= max_width:
            current_line.append(word)
        else:
            lines.append(" ".join(current_line))
            current_line = [word]
    lines.append(" ".join(current_line))

    y_offset = y_start
    for line in lines:
        rendered = font.render(line, True, color)
        rect = rendered.get_rect(center=(screen_width // 2, y_offset))
        surface.blit(rendered, rect)
        y_offset += font.get_linesize()
    return y_offset
