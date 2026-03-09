from datetime import datetime, timezone


def get_formatted_current_datetime():
    """
    Returns the current date and time formatted as "YYYY-MM-DD-HHMMSS".

    Returns:
      str: The formatted date and time string.
    """
    now = datetime.now(timezone.utc)
    formatted_datetime = now.strftime("%Y-%m-%d-%H%M%S")
    return formatted_datetime
