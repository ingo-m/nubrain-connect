import random
from copy import copy


def words_identical(word_1: str, word_2: str):
    """
    Compare words, ignoring capitalization and punctuation. For example "Horse" and
    "horse" are counted as identical. Similarly, "horse," and "horse" are considered
    identical. The reason for this is that in the listening condition it would
    otherwise be difficult to identify target events (because "horse," and "horse" sound
    similar).
    """
    word_1 = word_1.lower()
    word_2 = word_2.lower()

    word_1 = "".join(char for char in word_1 if char.isalnum())
    word_2 = "".join(char for char in word_2 if char.isalnum())

    if word_1 == word_2:
        return True
    else:
        return False


def get_target_events(*, text: list[str]):
    """
    Check for target events (i.e. repeated words) in text.
    """
    target_event_idcs = []

    for idx_word in range(0, (len(text) - 1)):
        this_word = text[idx_word]
        next_word = text[idx_word + 1]
        if words_identical(this_word, next_word):
            target_event_idcs.append(idx_word + 1)

    return target_event_idcs


def check_targets_too_close(*, target_idcs: list[int], min_distance_targets: int):
    """
    Ensure that two target events do not directly follow each other.
    """
    targets_too_close = None

    target_idcs = sorted(target_idcs)

    # Loop over target events.
    for idx_sample in range(0, (len(target_idcs) - 1)):
        # Current target event (word index, e.g. 7 would correspond to the 7th word in
        # the text.
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
        if (
            len(result) >= 2
            and words_identical(result[-1], word)
            and words_identical(result[-2], word)
        ):
            continue  # Skip adding this word

        result.append(word)

    return result


def sample_target_events(
    *,
    text: list[str],
    n_target_events: int,
    min_distance_targets: int = 3,
):
    n_words = len(text)
    if n_words <= (n_target_events * min_distance_targets * 2):
        raise AssertionError("Too many target events")

    # ----------------------------------------------------------------------------------
    # *** Process "naturally" occurring target events

    # Check for "naturally" occurring target events (i.e. repeated words) in the
    # original text.
    natural_target_event_idcs = get_target_events(text=text)
    n_natural_target_events = len(natural_target_event_idcs)
    print(
        f"Number of 'natural' target events (repeated words): {n_natural_target_events}"
    )

    # Remove potential double repeats (among 'natural' target events in the original
    # text).
    text = remove_double_repeats(text=text)

    # Check for "naturally" occurring target events (i.e. repeated words) in text after
    # removing double repeats.
    natural_target_event_idcs = get_target_events(text=text)
    n_natural_target_events = len(natural_target_event_idcs)
    print(
        "Number of 'natural' target events (repeated words) after removing potential "
        "double repeats: "
        f"{n_natural_target_events}"
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
                text[(natural_targets_too_close - 4) : (natural_targets_too_close + 4)]
            )
            print(
                "Natural target events too close, removing double target event: "
                f"{context}"
            )

            text.pop(natural_targets_too_close)  # Remove the xth word

            new_context = " ".join(
                text[(natural_targets_too_close - 4) : (natural_targets_too_close + 3)]
            )
            print(f"Context after removing double target event: {new_context}")

        else:
            done = True

    # Check for "naturally" occurring target events (i.e. repeated words) in text after
    # removing double target events.
    natural_target_event_idcs = get_target_events(text=text)
    n_natural_target_events = len(natural_target_event_idcs)
    print(
        "Number of 'natural' target events (repeated words) after removing potential "
        "double target events: "
        f"{n_natural_target_events}"
    )

    # ----------------------------------------------------------------------------------
    # *** Sample random target events

    # Update number of words after potentially removing natural / double target events.
    n_words = len(text)

    # Indices of target events (on a target event, the word will be repeated). For
    # example, a target event index of 7 means that the 7th word in the text will be a
    # target event (and will be repeated).

    done = False
    while not done:
        # Check if we need to add additional, randomly sampled target events (in
        # addition to potentially occurring "natural" target events in the original
        # text).
        if n_natural_target_events < n_target_events:
            # Inefficient solution, but doesn't matter as it only needs to run once,
            # offline.
            target_event_word_idcs = copy(natural_target_event_idcs)
            while len(target_event_word_idcs) < n_target_events:
                potential_sample = random.randrange(1, (n_words - 1))
                targets_too_close = check_targets_too_close(
                    target_idcs=copy(target_event_word_idcs) + [potential_sample],
                    min_distance_targets=min_distance_targets,
                )
                if targets_too_close:
                    continue
                else:
                    target_event_word_idcs.append(potential_sample)
        else:
            # There are enough 'natural' target events, no need to sample additional
            # ones.
            target_event_word_idcs = natural_target_event_idcs

        target_event_word_idcs = sorted(target_event_word_idcs)

        # Check whether the randomly sampled target events are too close to each other
        # or to the "natural" target events.
        targets_too_close = check_targets_too_close(
            target_idcs=target_event_word_idcs,
            min_distance_targets=min_distance_targets,
        )

        if targets_too_close:
            print("Target events too close. Will sample again.")
            continue
        else:
            # Targets are not too close, we can keep the randomly samples target event
            # indices.
            done = True
            break

    # ----------------------------------------------------------------------------------
    # *** Add the target events (repeated words) to the text

    # We need to add the randomly sampled target events (word repetitions) to the text,
    # but not the "naturally" occurring target events (because those are already in the
    # text).
    text_with_targets = []
    is_target = []  # Boolean list

    for idx_word, word in enumerate(text):
        if idx_word in natural_target_event_idcs:
            # The current word is a "natural" target event (i.e. a word repetition in
            # the original text).
            text_with_targets.append(word)
            is_target.append(True)
        elif idx_word in target_event_word_idcs:  # This has to be elif
            # This is a randomly sampled target event. Note that it is important that
            # this is not a "natural" target event, hence the `elif` after checking for
            # a "natural" target. Append the word twice (so that there is a repetition).
            text_with_targets.extend([word, word])
            # The second occurrence of the word is the target event.
            is_target.extend([False, True])
        else:
            # The current word is not a target event.
            text_with_targets.append(word)
            is_target.append(False)

    return {
        "text_with_targets": text_with_targets,
        "is_target": is_target,
    }
