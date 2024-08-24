import json
import logging
from rich.logging import RichHandler
from rich.console import Console
from rich.theme import Theme
from pathlib import Path

BASE_SDXL_MODEL_NAME = "sdxl_base_1.0_0.9_vae.safetensors"
BASE_FINE_TUNED_NAME = "epicrealism_v8.safetensors"

BASE_FLUX_DEV_MODEL_NAME = "flux1-dev.safetensors"
BASE_FLUX_DEV_CLIP_NAME = "clip_l.safetensors"
BASE_FLUX_DEV_T5_NAME = "t5_xxl_fp16.safetensors"
BASE_FLUX_DEV_AE_NAME = "ae.safetensors"


def are_models_verified_flux(log: logging.Logger) -> bool:
    # Verify that the correct Flux models are in the correct directories.

    models_dir = Path.cwd().joinpath("models", "flux_base_models")
    path_to_flux_dev_model = models_dir.joinpath(BASE_FLUX_DEV_MODEL_NAME)
    path_to_flux_dev_clip = models_dir.joinpath(BASE_FLUX_DEV_CLIP_NAME)
    path_to_flux_dev_t5 = models_dir.joinpath(BASE_FLUX_DEV_T5_NAME)
    path_to_flux_dev_ae = models_dir.joinpath(BASE_FLUX_DEV_AE_NAME)

    if not models_dir.exists():
        log.error("The Flux directory under models does not exist.")
        return False

    if not path_to_flux_dev_model.exists() and path_to_flux_dev_model.suffix == ".safetensors":
        log.error("The base Flux.1 [dev] model does not exist.")
        return False

    if not path_to_flux_dev_clip.exists() and path_to_flux_dev_clip.suffix == ".safetensors":
        log.error("The base Flux.1 [dev] clip_l model does not exist.")
        return False

    if not path_to_flux_dev_t5.exists() and path_to_flux_dev_t5.suffix == ".safetensors":
        log.error("The base Flux.1 [dev] t5 model does not exist.")
        return False

    if not path_to_flux_dev_ae.exists() and path_to_flux_dev_ae.suffix == ".safetensors":
        log.error("The base Flux.1 [dev] ae model does not exist.")
        return False

    log.info("All Flux models have been verified.")
    return True


def are_models_verified(log: logging.Logger) -> bool:
    # Verify that the correct models exist in the correct directory.

    models_dir = Path.cwd().joinpath("models")
    base_sdxl_file = models_dir.joinpath(BASE_SDXL_MODEL_NAME)
    base_fine_tuned_file = models_dir.joinpath(BASE_FINE_TUNED_NAME)

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
        log_time_format="%Y-%m-%d %H:%M:%S",
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
    pass
