class GlobalConfig:
    """
    Global config.
    """

    def __init__(self, version: str = "v1"):
        self.config_version = version
        self.v1.rest_condition_color = (128, 128, 128)
        self.v1.stim_start_marker = 1.0
        self.v1.stim_end_marker = 2.0
        self.v1.hdf5_dtype = "float32"
