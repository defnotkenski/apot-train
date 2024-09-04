import sys
import argparse
import zipfile
import utils
from pathlib import Path
import tempfile
import yaml
from utils import (
    setup_logging, accelerate_config_cmd, convert_to_toml_config, execute_cmd, is_finished_training, terminate_subprocesses,
    are_models_verified, notify_slack, upload_to_huggingface, get_executable_path, BASE_SDXL_MODEL_NAME, BASE_FINE_TUNED_NAME,
    SLACK_CHANNEL_ID
)

# TODO List ========================

# Done: Add lora extraction.
# Done: Add lora merging.
# Done: Add Huggingface support.
# Done: Add Flux.1 [dev] Lora support.

# TODO List ========================

# Set up logging.
log = setup_logging()

# Create temp output directory.
temp_output_dir = Path(tempfile.mkdtemp(prefix="outputs_"))

# Get the absolute path of the DIRECTORY containing THIS script.
script_dir = Path.cwd()
path_to_accelerate_executable = get_executable_path("accelerate")
path_to_accelerate_config = script_dir.joinpath("configs", "accelerate.yaml")
PYTHON = sys.executable
REPLICATE_REPO_ID = "notkenski/apothecary-dev"

# Insert SD_Scripts into PYTHONPATH.
sys.path.insert(0, str(script_dir.joinpath("sd_scripts")))


def setup_parser() -> argparse.ArgumentParser:
    # Set up and add arguments for the parser.

    parser = argparse.ArgumentParser()

    # General arguments
    parser.add_argument("--session_name", default=None, required=True, help="Name of this training session.")

    # Dreambooth arguments
    parser.add_argument("--train_dir", default=None, required=True, help="Path or training data in zip format.")
    parser.add_argument("--output_dir", default=None, required=True, help="Path of output directory.")
    parser.add_argument("--upload", default=None, help="Whether or not to upload to Huggingface Hub or save locally using token.")

    return parser


def train_sdxl(args: argparse.Namespace) -> None:
    # Begin actual training.

    # Create paths to files.
    base_sdxl_file_path = script_dir.joinpath("models", BASE_SDXL_MODEL_NAME)
    script_file_path = script_dir.joinpath("sd_scripts", "sdxl_train.py")
    path_to_sdxl_config = script_dir.joinpath("configs", "sdxl_dreambooth.yaml")

    # Extract zip file contents and empty into temp directory.
    temp_train_dir = tempfile.mkdtemp()
    with zipfile.ZipFile(args.train_dir, "r") as zip_ref:
        zip_ref.extractall(temp_train_dir)

    # Find the accelerate executable path.
    accelerate_path = utils.get_executable_path("accelerate")
    if accelerate_path == "":
        log.error("Accelerate executable not found.")
        return

    # Begin formulating the run command.
    run_cmd = [accelerate_path, "launch", "--config_file", str(path_to_accelerate_config)]
    run_cmd = accelerate_config_cmd(run_cmd=run_cmd)
    run_cmd.append(str(script_file_path))

    # Add TOML config argument.
    toml_config_path = convert_to_toml_config(config_path=str(path_to_sdxl_config))
    run_cmd.append("--config_file")
    run_cmd.append(toml_config_path)

    # Add extra SDXL script arguments.
    run_cmd.append("--train_data_dir")
    run_cmd.append(temp_train_dir)
    run_cmd.append("--pretrained_model_name_or_path")
    run_cmd.append(str(base_sdxl_file_path))
    run_cmd.append("--output_dir")
    run_cmd.append(str(temp_output_dir))
    run_cmd.append("--output_name")
    run_cmd.append(f"{args.session_name}_dreambooth")

    # Execute the command.
    executed_subprocess = execute_cmd(run_cmd=run_cmd, log=log)

    # Check to see if subprocess is finished yet.
    is_finished_training(executed_subprocess, log=log)

    # Once finished, make sure that all subprocesses are terminated after completion.
    terminate_subprocesses(executed_subprocess, log=log)

    return


def extract_lora(args: argparse.Namespace) -> None:
    # Extract lora from trained SDXL model.

    # Load lora extraction config into variable.
    with open("configs/sdxl_xlora.yaml", "r") as read_xlora:
        xlora_config = yaml.safe_load(read_xlora)

    cleaned_xlora_config = {
        key: xlora_config[key] for key in xlora_config if xlora_config[key] not in [""]
    }

    # Create paths to appropriate files.
    path_to_script = script_dir.joinpath("sd_scripts", "networks", "extract_lora_from_models.py")
    base_sdxl_file_path = script_dir.joinpath("models", BASE_SDXL_MODEL_NAME)
    dreambooth_file_path = temp_output_dir.joinpath(f"{args.session_name}_dreambooth.safetensors")
    save_to_path = temp_output_dir.joinpath(f"{args.session_name}_xlora.safetensors")

    # Establish argument paths in run command.
    run_cmd = [
        path_to_accelerate_executable,
        "launch",
        "--config_file",
        str(path_to_accelerate_config),
        str(path_to_script),
        "--model_org",
        str(base_sdxl_file_path),
        "--model_tuned",
        str(dreambooth_file_path),
        "--save_to",
        str(save_to_path),
    ]

    for key in cleaned_xlora_config:
        if key not in ["model_org", "model_tuned", "save_to"]:
            run_cmd.append(f"--{key}")
            if cleaned_xlora_config[key] is not True:
                run_cmd.append(str(cleaned_xlora_config[key]))

    # Execute the command.
    executed_subprocess = execute_cmd(run_cmd=run_cmd, log=log)

    # Check to see if the subprocess has finished yet.
    is_finished_training(executed_subprocess, log=log)

    # Once completed, make sure all processes are terminated.
    terminate_subprocesses(executed_subprocess, log=log)

    return


def merge_lora(args: argparse.PARSER) -> None:
    # Merge lora into SDXL base fine-tuned model.

    # Load JSON configuration to a dictionary.
    with open("configs/sdxl_mlora.yaml", "r") as read_xlora_config:
        mlora_config = yaml.safe_load(read_xlora_config)

    # Remove unset lines in configuration.
    cleaned_mlora_config = {
        key: mlora_config[key] for key in mlora_config if mlora_config[key] not in [""]
    }

    # Create appropriate paths for files.
    path_to_script = script_dir.joinpath("sd_scripts", "networks", "sdxl_merge_lora.py")
    base_fine_tuned_model = script_dir.joinpath("models", BASE_FINE_TUNED_NAME)
    extracted_lora_model = temp_output_dir.joinpath(f"{args.session_name}_xlora.safetensors")
    output_path = Path(args.output_dir).joinpath(f"{args.session_name}_final.safetensors")

    # Create the run command to be executed with paths as the foundation.
    run_cmd = [
        path_to_accelerate_executable,
        "launch",
        "--config_file",
        str(path_to_accelerate_config),
        str(path_to_script),
        rf"--sd_model",
        str(base_fine_tuned_model),
        rf"--model",
        str(extracted_lora_model),
        rf"--save_to",
        str(output_path)
    ]

    # Append addition arguments to the run command from JSON configuration.
    for key in cleaned_mlora_config:
        if key not in ["sd_model", "model", "save_to"]:
            run_cmd.append(rf"--{key}")

            if cleaned_mlora_config[key] is not True:
                run_cmd.append(str(cleaned_mlora_config[key]))

    # Execute the command.
    executed_subprocess = execute_cmd(run_cmd=run_cmd, log=log)

    # Check to see if subprocess is still running.
    is_finished_training(executed_subprocess, log=log)

    # Make sure all subprocesses are gone before continuing.
    terminate_subprocesses(executed_subprocess, log=log)

    return


if __name__ == "__main__":
    # Set up parser for CLI.
    configured_parser = setup_parser()
    train_args = configured_parser.parse_args()

    # Now begin training pipeline.
    log.info("[reverse wheat1]Starting training session.", extra={"markup": True})

    # Check if the correct base models are in the models directory.
    if not are_models_verified(log):
        sys.exit()

    # Check if the temp output directory exists.
    if not temp_output_dir.exists():
        log.error("Temporary output directory was not established.")
        sys.exit()
    else:
        log.info("Temporary output directory verified.")

    # Begin training script executions.
    log.info("[reverse wheat1]Starting Dreambooth training.", extra={"markup": True})
    train_sdxl(args=train_args)

    log.info("[reverse wheat1]Starting lora extraction.", extra={"markup": True})
    extract_lora(args=train_args)

    log.info("[reverse wheat1]Starting lora merging.", extra={"markup": True})
    merge_lora(args=train_args)

    # Upload file to Huggingface Hub if set in CLI.
    path_to_final_model = Path(train_args.output_dir).joinpath(f"{train_args.session_name}_final.safetensors")
    path_to_yaml_upload = script_dir.joinpath("configs", "sdxl_dreambooth.yaml")
    upload_to_huggingface(model_path=path_to_final_model, yaml_path=path_to_yaml_upload, log=log, train_args=train_args)

    # Training session complete.
    log.info("[reverse honeydew2]Training session is now complete.", extra={"markup": True})

    if train_args.notify is not None:
        notification_message = f"Dreambooth training has completed for {train_args.session_name} ✨🦖"
        notify_slack(channel_id=SLACK_CHANNEL_ID, msg=notification_message, log=log, train_args=train_args)
