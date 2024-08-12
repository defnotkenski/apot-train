import shutil
import os
import subprocess
import sys
import argparse
import time
import zipfile
import tempfile
import json
import psutil
import toml
from utils import setup_logging

# TODO List ========================

# DONE: Automatically configure accelerate config.
# DONE: Upload JSON config file instead of TOML (wtf).
# DONE: Capture when subprocess is done (Maybe polling can work?).
# DONE: See if a temp output can work.

# TODO List ========================

# Set up logging
log = setup_logging()

# Get the absolute path of the DIRECTORY containing THIS script
script_dir = os.path.dirname(os.path.abspath(__file__))

# Insert SD_Scripts into PYTHONPATH
sys.path.insert(0, os.path.join(script_dir, "sd_scripts"))


def setup_parser() -> argparse.ArgumentParser:
    # Set up and add arguments for the parser

    parser = argparse.ArgumentParser()

    parser.add_argument("--json_config", default=None, help="JSON configuration file path.")
    parser.add_argument("--train_data_zip", default=None, help="Path or training data in zip format.")
    parser.add_argument("--output_dir", default=None, help="Path of output directory.")

    return parser


def get_executable_path(name: str) -> str:
    # Get path for accelerate executable.

    executable_path = shutil.which(name)
    if executable_path is None:
        return ""

    return executable_path


def accelerate_config_cmd(run_cmd: list) -> list:
    # Lay out accelerate arguments for the run command.

    run_cmd.append("--dynamo_backend")
    run_cmd.append("no")

    run_cmd.append("--dynamo_mode")
    run_cmd.append("default")

    run_cmd.append("--mixed_precision")
    run_cmd.append("fp16")

    run_cmd.append("--num_processes")
    run_cmd.append("1")

    run_cmd.append("--num_machines")
    run_cmd.append("1")

    run_cmd.append("--num_cpu_threads_per_process")
    run_cmd.append("2")

    return run_cmd


def execute_cmd(run_cmd: list) -> subprocess.Popen:
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


def begin_json_config(config_path) -> str:
    # Remove blank lines from JSON config and convert to a TOML file

    _, tmp_toml_path = tempfile.mkstemp(suffix=".toml")

    with open(config_path, "r") as json_config_read, open(tmp_toml_path, "w", encoding="utf-8") as toml_write:
        json_dict = json.load(json_config_read)
        # print(json_dict)

        cleaned_json_dict = {
            key: json_dict[key] for key in json_dict if json_dict[key] not in [""]
        }
        # print(f"Parsed JSON:\n{cleaned_json_dict}")

        toml.dump(cleaned_json_dict, toml_write)

    return tmp_toml_path


def is_finished_training(process: subprocess.Popen) -> None:
    # Continuously check if subprocesses are finished

    while process.poll() is None:
        time.sleep(2)

    log.info("Training has ended.")

    return


def terminate_subprocesses(process: subprocess.Popen) -> None:
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


def train_sdxl(args) -> None:
    # Begin actual training

    # Extract zip file contents and empty into temp directory
    train_data_dir = tempfile.mkdtemp()
    with zipfile.ZipFile(args.train_data_zip, "r") as zip_ref:
        zip_ref.extractall(train_data_dir)

    # Find the accelerate executable path
    accelerate_path = get_executable_path("accelerate")
    if accelerate_path == "":
        log.error("Accelerate executable not found.")
        return

    run_cmd = [f"{accelerate_path}", "launch"]

    run_cmd = accelerate_config_cmd(run_cmd=run_cmd)

    run_cmd.append(rf"{script_dir}/sd_scripts/sdxl_train.py")

    # Add TOML config argument
    toml_config_path = begin_json_config(rf"{args.json_config}")
    run_cmd.append("--config_file")
    run_cmd.append(rf"{toml_config_path}")

    # Add extra SDXL script arguments
    run_cmd.append("--train_data_dir")
    run_cmd.append(rf"{train_data_dir}")
    run_cmd.append("--output_dir")
    run_cmd.append(rf"{args.output_dir}")

    executed_subprocess = execute_cmd(run_cmd=run_cmd)

    # Check to see if subprocess is finished yet
    is_finished_training(executed_subprocess)

    # Once finished, make sure that all subprocesses are terminated after completion
    terminate_subprocesses(executed_subprocess)

    return


if __name__ == "__main__":
    # Set up parser for CLI
    configured_parser = setup_parser()
    parsed_args = configured_parser.parse_args()

    # Now, begin inference
    log.info("Starting training for SDXL Dreambooth niggaaa!")
    train_sdxl(args=parsed_args)
