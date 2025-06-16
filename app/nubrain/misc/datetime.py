import datetime


def get_formatted_current_datetime():
    """
    Returns the current date and time formatted as "YYYY-MM-DD-HHMMSS".

    Returns:
      str: The formatted date and time string.
    """
    now = datetime.datetime.now()
    formatted_datetime = now.strftime("%Y-%m-%d-%H%M%S")
    return formatted_datetime
