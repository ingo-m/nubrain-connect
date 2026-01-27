import random


def get_target_events(*, text: list[str]):
    """
    Check for target events (i.e. repeated words) in text.
    """
    target_event_idcs = []

    for idx_word in range(0, (len(text) - 1)):
        this_word = text[idx_word]
        next_word = text[idx_word + 1]
        if this_word == next_word:
            target_event_idcs.append(idx_word)

    return target_event_idcs


def check_targets_too_close(*, target_idcs: list[int], min_distance_targets: int):
    """
    Ensure that two target events do not directly follow each other.
    """
    targets_too_close = None

    # Loop over target events.
    for idx_sample in range(len(target_idcs) - min_distance_targets):
        # Current target event (word index, e.g. 7 would correspond to the 7th word
        # in the text.
        target_idx = target_idcs[idx_sample]
        # Index of the subsequent target event.
        next_target_idx = target_idcs[idx_sample + 1]
        target_events_distance = next_target_idx - target_idx
        if target_events_distance < min_distance_targets:
            targets_too_close = target_idx
            break

    # Returns `None` or the integer index of a target (e.g. integer 7 if the 7th word in
    # the text is a target event) if that target event is followed by another target
    # event.
    return targets_too_close


def remove_double_repeats(*, text: list[str]):
    """
    Remove double repeated words (keep max 2 consecutive duplicates). For example, the
    input ["Apple", "Banana", "Banana", "Banana", "Coconut"] will get changed to
    ["Apple", "Banana", "Banana", "Coconut"]. In contrast, the input ["Apple", "Banana",
    "Banana", "Coconut"] will remained unchanged.
    """
    result = []

    for word in text:
        # Check if we already have at least 2 words in the result and if the last two
        # words match the current word.
        if len(result) >= 2 and result[-1] == word and result[-2] == word:
            continue  # Skip adding this word

        result.append(word)

    return result


def sample_target_events(
    *,
    text: list[str],
    n_words_to_show: int,
    n_target_events: int,
    min_distance_targets: int = 3,
):
    if n_words_to_show <= (n_target_events * min_distance_targets * 2):
        raise AssertionError("Too many target events")

    # Check for "naturally" occuring target events (i.e. repeated words) in text.
    natural_target_event_idcs = get_target_events(text=text)
    n_natural_target_event_idcs = len(natural_target_event_idcs)
    print(
        "Number of 'natural' target events (repeated words): "
        f"{n_natural_target_event_idcs}"
    )

    # Remove potential double repeats (among 'natural' target events in the original
    # text).
    text = remove_double_repeats(text=text)

    # Check for "naturally" occuring target events (i.e. repeated words) in text after
    # removing double repeats.
    natural_target_event_idcs = get_target_events(text=text)
    n_natural_target_event_idcs = len(natural_target_event_idcs)
    print(
        "Number of 'natural' target events (repeated words) after removing potential "
        "double repeats: "
        f"{n_natural_target_event_idcs}"
    )

    # If necessary, remove double target events (note that previously, words that
    # repeated more than once were removed; e.g. "Apple Banana Banana Banana Coconut"
    # became "Apple Banana Banana Coconut"; but now, "Apple Banana Banana Coconut
    # Coconut Kiwi" becomes "Apple Banana Banana Coconut Kiwi".
    done = False
    while not done:
        natural_targets_too_close = check_targets_too_close(
            target_idcs=get_target_events(text=text),
            min_distance_targets=min_distance_targets,
        )

        if natural_targets_too_close is not None:
            context = " ".join(
                text[
                    (natural_targets_too_close - 10) : (natural_targets_too_close + 10)
                ]
            )
            print(
                "Natural target events too close, removing double target event: "
                f"{context}"
            )

            text.pop(natural_targets_too_close)  # Remove the xth word

            new_context = " ".join(
                text[(natural_targets_too_close - 10) : (natural_targets_too_close + 9)]
            )
            print(f"Context after removing double target event: {new_context}")

        else:
            done = True

    # Check for "naturally" occuring target events (i.e. repeated words) in text after
    # removing double target events.
    natural_target_event_idcs = get_target_events(text=text)
    n_natural_target_event_idcs = len(natural_target_event_idcs)
    print(
        "Number of 'natural' target events (repeated words) after removing potential "
        "double target events: "
        f"{n_natural_target_event_idcs}"
    )

    done = False
    while not done:
        if n_natural_target_event_idcs < n_target_events:
            # Indices of target events (on a target event, the word will be repeated).
            # For example, a target event index of 7 means that the 7th word in the text
            # will be a target event (and will be repeated).
            target_event_word_idcs = random.sample(
                range(0, n_words_to_show),
                (n_target_events - n_natural_target_event_idcs),
            )
            target_event_word_idcs = target_event_word_idcs + natural_target_event_idcs
        else:
            # There are enough 'natural' target events, no need to sample additional
            # ones.
            target_event_word_idcs = natural_target_event_idcs

        target_event_word_idcs = sorted(target_event_word_idcs)

        targets_too_close = check_targets_too_close(
            target_idcs=target_event_word_idcs,
            min_distance_targets=min_distance_targets,
        )

        if targets_too_close:
            print(f"Target events too close: {targets_too_close} | Will sample again.")
            continue
        else:
            # Targets are not too close, we can keep the randomly samples target event
            # indices.
            done = True
            break

    return target_event_word_idcs
