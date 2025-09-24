class GlobalConfig:
    """
    Global config.
    """

    def __init__(self, version: str = "v1"):
        self.config_version = version
        # Color values for experimental rest condition (e.g. grey).
        self.rest_condition_color = (128, 128, 128)
        # Markers for stimulus start and end (will be stored in marker channel).
        self.stim_start_marker = 1.0
        self.stim_end_marker = 2.0
        # Data type for EEG data to use when saving to hdf5 file.
        self.hdf5_dtype = "float64"
        # Resize longest image dimension to this size when saving image to hdf5 file.
        self.max_img_storage_dimension = 128
