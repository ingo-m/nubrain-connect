from nubrain.global_config import GlobalConfig


class TextConfig:
    def __init__(self, version: str = "v1"):
        global_config = GlobalConfig()

        self.config_version = version
        # Color values for experimental rest condition (e.g. grey).
        self.rest_condition_color = (0, 0, 0)
        # Font colors. If more than one, will sample randomly on each trial. Will use
        # first color in list for rendering behavioural results.
        self.font_colors = [
            (255, 255, 255),
            (255, 0, 0),
            (0, 255, 0),
            (0, 0, 255),
            (0, 255, 255),
            (255, 0, 255),
            (255, 255, 0),
        ]
        # Markers for stimulus start and end (will be stored in marker channel).
        self.stim_start_marker = global_config.stim_start_marker
        self.stim_end_marker = global_config.stim_end_marker
        # Data type for EEG data to use when saving to hdf5 file.
        self.hdf5_dtype = global_config.hdf5_dtype
