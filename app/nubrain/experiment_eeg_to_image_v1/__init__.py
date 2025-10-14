from . import load_config, tone

# Wrap the main import in try, so that the other modules can be imported without
# dependency on pylsl (e.g. when using global_config during preprocessing).
try:
    from . import main
except Exception as e:
    print(f"Failed to import nubrain experiment_eeg_to_image_v1 main module: {e}")
