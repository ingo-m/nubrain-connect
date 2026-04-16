def map_session_config_to_experiment_config(
    *,
    session_config: dict,
    experiment_config: dict,
):
    """
    Map session config to experiment config. Specifically:
    - subject (integer), e.g. 1 -> subject_id (string), e.g. "sub-001"
    - session (integer), e.g. 1 -> session_id (string), e.g. "session-001"
    - run (integer), e.g. 1 -> section_idx_start (intger) = n_sections_to_show * (run - 1)
    """
    subject = session_config["subject"]
    session = session_config["session"]
    run = session_config["run"]

    subject_id = f"sub-{subject:03}"
    session_id = f"session-{session:03}"

    n_sections_to_show = experiment_config["n_sections_to_show"]
    # Minus one because we start at section zero on first run.
    section_idx_start = n_sections_to_show * (run - 1)

    experiment_config["subject_id"] = subject_id
    experiment_config["session_id"] = session_id
    experiment_config["section_idx_start"] = section_idx_start

    return experiment_config
