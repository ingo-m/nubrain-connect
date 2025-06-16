class GlobalConfig:
    """
    Global config.
    """

    def __init__(self, version: str = "v1"):
        self.config_version = version
        # Color values for experimental rest condition (e.g. grey).
        self.v1.rest_condition_color = (128, 128, 128)
        # Markers for stimulus start and end (will be stored in marker channel).
        self.v1.stim_start_marker = 1.0
        self.v1.stim_end_marker = 2.0
        # Data type for board data to use when saving to hdf5 file.
        self.v1.hdf5_dtype = "float32"
        # Resize longest image dimension to this size when saving image to hdf5 file.
        max_img_storage_dimension = 384
