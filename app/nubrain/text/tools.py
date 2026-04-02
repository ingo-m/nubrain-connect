def load_text(*, path_text: str):
    with open(path_text, "r", encoding="utf-8") as file:
        text = file.read()
    return text


class PunctuationChars:
    def __init__(self):
        self.allowed = set(" ,-.:;“”'‘’()!?…" + '"')


def exclude_invalid_chars(*, text: str):
    text = text.replace("\n", " ")
    text = text.replace("—", " ")
    text = text.replace("_", " ")
    text = text.replace(b"\xe3\x80\x80".decode("utf8"), " ")

    punctuation_chars = PunctuationChars().allowed

    alphanumeric_chars_allowed = set(
        "0123456789"
        + "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
        + "abcdefghijklmnopqrstuvwxyz"
        + "ÁÂÃÄÅÆÇÈÉÊËÍÎÏÐÑÒÓÔÕÖØÙÚÛÜÝ"
        + "ÞßàáâãäåæçèéêëíîïðñòóôõöøùúûüýþÿŌōŞşŪūŸŻż"
    )

    chars_allowed = alphanumeric_chars_allowed | punctuation_chars

    text_filtered = []

    for idx_char in range(len(text)):
        char = text[idx_char]
        if char not in chars_allowed:
            # Remove invalid character.
            try:
                context = text[(idx_char - 10) : (idx_char + 10)]
            except KeyError:
                context = None
            print(f"Removing char: {char} | Context: {context}")
        else:
            # Add valid character to filtered text.
            text_filtered.append(char)

    text_filtered = "".join(text_filtered)

    return text_filtered


def load_and_preprocess_text(*, path_text: str):
    text = load_text(path_text=path_text)

    text = exclude_invalid_chars(text=text)

    # One large string to list of words.
    text = text.split(" ")

    # Remove empty strings.
    text = [x for x in text if len(x) > 0]

    return text
