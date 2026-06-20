from __future__ import annotations

import logging
from pathlib import Path


def configure_logging(
        name: str, 
        log_file: Path, 
        level=logging.INFO,
        overwrite_handlers: bool = False,
    ) -> logging.Logger:
    """
    Configure a named logger with console and file handlers.

    Parameters:
        name: Logger name to configure.
        log_file: File path where log records should be written.
        level: Logging threshold applied to the named logger.
        overwrite_handlers: When True, clear existing handlers before setup.

    Returns:
        Configured logger instance.

    Side effects:
        Creates the log file parent directory and attaches stream/file handlers
        when the logger does not already have handlers.

    Typical usage:
        logger = configure_logging("capstone", paths.logs / "silver.log")
        child = logging.getLogger("capstone.paths")  # inherits handlers from capstone

    Notes:
    - Intended to be called once per process/kernel.
    - Notebook-safe: prevents duplicate handlers unless overwrite_handlers=True.
    """

    log_file.parent.mkdir(parents=True, exist_ok=True)

    logger = logging.getLogger(name)
    logger.setLevel(level)

    # False to prevent double logging, child will still propagate 
    logger.propagate = False

    if overwrite_handlers:
        logger.handlers.clear()

    if not logger.handlers:
        
        formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(name)s | %(message)s")

        streamhandler = logging.StreamHandler()
        streamhandler.setFormatter(formatter)

        filehandler = logging.FileHandler(log_file, encoding="utf-8")
        filehandler.setFormatter(formatter)

        logger.addHandler(streamhandler)
        logger.addHandler(filehandler)

    return logger


# https://docs.python.org/3/howto/logging.html
# https://docs.python.org/3/library/logging.handlers.html
# https://stackoverflow.com/questions/7173033/duplicate-log-output-when-using-python-logging-module


#### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### 
#### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### 

def log_layer_paths(
    paths, 
    current_layer: str, 
    logger: logging.Logger
    ) -> None:

    """
    Log common project paths, the current layer paths, and the previous layer paths
    when applicable.

    Parameters:
        paths: Object containing project path attributes such as ``root`` and
            ``data_silver``.
        current_layer: Active medallion layer name. Must be one of bronze,
            silver, gold, or synthetic.
        logger: Logger used to emit discovered path values.

    Returns:
        None.

    Raises:
        ValueError: If ``current_layer`` is not a supported layer.

    Side effects:
        Emits info-level log messages for available path attributes.

    Example:
        log_layer_paths(paths, current_layer="silver", logger=logger)
    """
    layer = current_layer.strip().lower()

    valid_layers = ["bronze", "silver", "gold", "synthetic"]
    if layer not in valid_layers:
        raise ValueError(f"current_layer must be one of {valid_layers}, got '{current_layer}'")

    current_index = valid_layers.index(layer)
    # Guard against index 0: without it, valid_layers[-1] would wrap to "synthetic",
    # giving Bronze a wrong previous-layer value due to Python's negative index behavior.
    previous_layer = valid_layers[current_index - 1] if current_index > 0 else None

    common_paths = [
        ("Project Root Path Loaded", "root"),
        ("Project Logging Path Loaded", "logs"),
        ("Project Artifacts Path Loaded", "artifacts"),
        ("Project Notebooks Path Loaded", "notebooks"),
        ("Project Truths Path Loaded", "truths"),
        ("Project Data Path Loaded", "data"),
    ]

    current_layer_paths = [
        (f"Data {layer.capitalize()} Path Loaded", f"data_{layer}"),
        (f"Data {layer.capitalize()} Training Path Loaded", f"data_{layer}_train"),
        (f"Data {layer.capitalize()} Testing Path Loaded", f"data_{layer}_test"),
    ]

    previous_layer_paths = []
    if previous_layer is not None:
        previous_layer_paths = [
            (f"Previous Layer ({previous_layer.capitalize()}) Path Loaded", f"data_{previous_layer}"),
            (
                f"Previous Layer ({previous_layer.capitalize()}) Training Path Loaded",
                f"data_{previous_layer}_train",
            ),
            (
                f"Previous Layer ({previous_layer.capitalize()}) Testing Path Loaded",
                f"data_{previous_layer}_test",
            ),
        ]

    for message, attribute_name in common_paths + previous_layer_paths + current_layer_paths:
        if hasattr(paths, attribute_name):
            logger.info("%s: %s", message, getattr(paths, attribute_name))


#### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### 
#### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### 
