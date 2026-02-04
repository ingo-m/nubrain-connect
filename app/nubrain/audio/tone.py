import numpy as np


def generate_tone(
    *,
    frequency: int,
    duration: float,
    amplitude: float,
    sample_rate: int,
):
    """Generates a numpy array containing a sine wave for a pure tone."""
    # Calculate the number of samples needed.
    n_samples = int(round(sample_rate * duration))

    # Create an array representing time points.
    time_points = np.linspace(0, duration, n_samples, endpoint=False)

    # Generate the sine wave.
    # 2 * pi * frequency * t gives the angle for the sine function.
    wave = np.sin(2 * np.pi * frequency * time_points)

    # Scale wave to 16-bit integer range (-32767 to 32767) and apply amplitude. This
    # format is required by Pygame's mixer.
    wave = np.clip(
        np.around(wave * amplitude * 32767.0),
        a_min=-32767.0,
        a_max=32767.0,
    ).astype(np.int16)

    # Create a 2D array for stereo sound by duplicating the mono wave. Pygame's sndarray
    # requires a (num_samples, 2) array for stereo.
    stereo_wave = np.column_stack((wave, wave))

    return stereo_wave
