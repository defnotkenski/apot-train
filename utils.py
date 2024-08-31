import argparse
import time
import tempfile
import toml
import json
import logging
from rich.logging import RichHandler
from rich.console import Console
from rich.theme import Theme
from pathlib import Path
import shutil
import subprocess
import yaml
import psutil
from huggingface_hub import HfApi
from slack_sdk import WebClient
from slack_sdk.errors import SlackApiError

BASE_SDXL_MODEL_NAME = "sdxl_base_1.0_0.9_vae.safetensors"
BASE_FINE_TUNED_NAME = "epicrealism_v8.safetensors"

BASE_FLUX_DEV_MODEL_NAME = "flux1-dev.safetensors"
BASE_FLUX_DEV_CLIP_NAME = "clip_l.safetensors"
BASE_FLUX_DEV_T5_NAME = "t5xxl_fp16.safetensors"
BASE_FLUX_DEV_AE_NAME = "ae.safetensors"
REPLICATE_REPO_ID = "notkenski/apothecary-dev"

SLACK_TOKEN = ""


def notify_slack(channel_id: str, msg: str, log: logging.Logger) -> None:
    # Notify a specific Slack channel on training updates.

    client = WebClient(token=SLACK_TOKEN)

    try:
        response = client.chat_postMessage(
            channel=channel_id,
            text=msg
        )
        log.debug(response)
    except SlackApiError as e:
        log.error(f"Error with slack: {e}")


def upload_to_huggingface(model_path: Path, log: logging.Logger, train_args: argparse.PARSER) -> None:
    # Upload provided model to the Huggingface Repository.

    try:
        if train_args.upload is not None:
            log.info("[reverse wheat1]Starting upload to Huggingface Hub.", extra={"markup": True})

            hf_api = HfApi()
            hf_api.upload_file(
                token=train_args.upload,
                path_or_fileobj=model_path,
                path_in_repo=model_path.name,
                repo_id=REPLICATE_REPO_ID
            )
    except Exception as e:
        log.error(f"Exception during Huggingface upload: {e}")


def terminate_subprocesses(process: subprocess.Popen, log: logging.Logger) -> None:
    # Kill all processes that are currently running

    if process.poll() is None:
        try:
            parent = psutil.Process(process.pid)
            for child in parent.children(recursive=True):
                child.kill()
            parent.kill()
            log.info("Running process has been killed.")
        except psutil.NoSuchProcess:
            log.info("This process does not exist anymore.")
        except Exception as e:
            log.error(f"Error terminating process: {e}")
    else:
        log.info("There is no process to kill.")

    return


def is_finished_training(process: subprocess.Popen, log: logging.Logger) -> None:
    # Continuously check if subprocesses are finished

    while process.poll() is None:
        time.sleep(2)

    log.info("Training has ended.")

    return


def convert_to_toml_config(config_path: str) -> str:
    # Remove blank lines from YAML config and convert to a TOML file

    _, tmp_toml_path = tempfile.mkstemp(suffix=".toml")

    with open(config_path, "r") as config_read, open(tmp_toml_path, "w", encoding="utf-8") as toml_write:
        config_dict = yaml.safe_load(config_read)
        # log.debug(config_dict)

        cleaned_dict = {
            key: config_dict[key] for key in config_dict if config_dict[key] not in [""]
        }
        # log.debug(cleaned_dict)

        toml.dump(cleaned_dict, toml_write)

    return tmp_toml_path


def execute_cmd(run_cmd: list[str], log: logging.Logger) -> subprocess.Popen:
    # Execute the training command

    # Reformat for user friendly display
    command_to_run = " ".join(run_cmd)
    log.info(f"Executing command: {command_to_run}")

    # Execute the command
    process = subprocess.Popen(run_cmd)

    # while True:
    #     line = process.stdout.readline()
    #     if not line and process.poll() is not None:
    #         break
    #     print(line.decode(), end="")

    # Remember, a return of None means the child process has not been terminated (wtf)
    if process.poll() is not None:
        log.error("Command could not be executed.")

    log.info("Command executed & running.")
    return process


def accelerate_config_cmd(run_cmd: list) -> list:
    # Lay out accelerate arguments for the run command.

    # run_cmd.append("--dynamo_backend")
    # run_cmd.append("no")

    # run_cmd.append("--dynamo_mode")
    # run_cmd.append("default")

    # run_cmd.append("--mixed_precision")
    # run_cmd.append("bf16")

    # run_cmd.append("--num_processes")
    # run_cmd.append("1")

    # run_cmd.append("--num_machines")
    # run_cmd.append("1")

    # run_cmd.append("--num_cpu_threads_per_process")
    # run_cmd.append("2")

    return run_cmd


def get_executable_path(name: str) -> str:
    # Get path for accelerate executable.

    executable_path = shutil.which(name)
    if executable_path is None:
        return ""

    return executable_path


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
