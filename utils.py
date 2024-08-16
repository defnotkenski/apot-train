import json
import logging
from rich.logging import RichHandler
from rich.console import Console
from rich.theme import Theme
from pathlib import Path

BASE_SDXL_MODEL_NAME = "sdxl_base_1.0_0.9_vae.safetensors"
BASE_FINE_TUNED = "epicrealism_v8.safetensors"


def are_models_verified(log: logging.Logger) -> bool:
    # Verify that the correct models exist in the correct directory.

    models_dir = Path.cwd().joinpath("models")
    base_sdxl_file = models_dir.joinpath(BASE_SDXL_MODEL_NAME)
    base_fine_tuned_file = models_dir.joinpath(BASE_FINE_TUNED)

    if not models_dir.exists():
        log.error("Models directory does not exist.")
        return False

    if not base_sdxl_file.exists() and base_sdxl_file.suffix == ".safetensors":
        log.error("Check your base SDXL file in models directory does not exist.")
        return False

    if not base_fine_tuned_file.exists() and base_fine_tuned_file.suffix == ".safetensors":
        log.error("Check your base fine-tuned file in models directory.")
        return False

    log.info("Base models have been verified.")
    return True


def setup_logging() -> logging.Logger:
    # Set up the logger with pertyyy Rich logging

    rich_console = Console(stderr=True, theme=Theme({
        "log.time": "dim magenta",
        "logging.level.debug": "bold cyan1",
        "logging.level.info": "bold cyan1",
        "logging.level.warning": "bold yellow1",
        "logging.level.error": "bold yellow1",
        "logging.level.critical": "bold reverse yellow1",
        "log.message": "pink1"
    }))

    rh = RichHandler(
        show_time=True,
        omit_repeated_times=False,
        show_level=True,
        show_path=False,
        markup=False,
        rich_tracebacks=True,
        log_time_format="[ %Y-%m-%d %H:%M:%S ]",
        level=logging.DEBUG,
        console=rich_console,
    )

    logger = logging.getLogger("banana_nut")

    if logger.hasHandlers():
        logger.handlers.clear()

    logger.addHandler(rh)
    logger.setLevel(logging.DEBUG)

    return logger


def sort_json(json_path, output_name):
    # Sort a JSON file containing key, value pairs based on keys

    with open(f"{json_path}", "r") as json_read, open(f"{output_name}.json", "w") as json_write:
        json_dict = json.load(json_read)
        json.dump({k: json_dict[k] for k in sorted(json_dict)}, json_write, indent=2)

    return f"Saved file as {output_name}.json"


if __name__ == "__main__":
    # Sorting JSON file by keyname
    # print(sort_json("config_dreambooth.json", "config_dreambooth"))

    pass
