import argparse
import sys
from utils import setup_logging, BASE_FLUX_DEV_MODEL_NAME, are_models_verified_flux, BASE_FLUX_DEV_CLIP_NAME, BASE_FLUX_DEV_T5_NAME, BASE_FLUX_DEV_AE_NAME
from pathlib import Path
import tempfile
import zipfile
from main_sdxl import get_executable_path, accelerate_config_cmd, begin_json_config, execute_cmd, is_finished_training, terminate_subprocesses

# Some variable setups to be commonly used throughout this script. Varibles in UPPERCASE are subject to change by the user.
log = setup_logging()
script_dir = Path.cwd()
python = sys.executable

# Add sd_scripts_flux submodule to python's path.
sys.path.insert(0, str(script_dir.joinpath("sd_scripts_flux")))


def setup_parser() -> argparse.ArgumentParser:
    # Set up the parser to accept CLI arguments.

    parser = argparse.ArgumentParser()
    parser.add_argument("--session_name", default=None, required=True, help="Name of this training session (Will appear as file names).")
    parser.add_argument("--training_data_zip", default=None, required=True, help="Path of training data in zip format.")
    parser.add_argument("--output_dir", default=None, required=True, help="Path to the local output directory.")

    # Automatically set, but can be user-defined in the CLI.
    # parser.add_argument("--flux_config", default=None, required=True, help="Configuration JSON file for Flux training.")
    # parser.add_argument("--clip_l", default=None, required=True, help="Path to the clip_l model.")
    # parser.add_argument("--t5xxl", default=None, required=True, help="Path to the t5xxl model.")
    # parser.add_argument("--ae", default=None, required=True, help="Path to the ae model.")

    return parser


def train_flux(args: argparse.Namespace) -> None:
    # Begin training of the Flux model Lora.

    # Create appropriate paths to files.
    path_to_script = script_dir.joinpath("sd_scripts_flux", "flux_train_network.py")
    path_to_accelerate_config = script_dir.joinpath("configs", "accelerate.yaml")
    path_to_flux_config = script_dir.joinpath("configs", "flux_lora.json")

    # Unzip file and store in temp directory.
    temp_train_dir = tempfile.mkdtemp()
    with zipfile.ZipFile(args.training_data_zip, "r") as zip_ref:
        zip_ref.extractall(temp_train_dir)

    # Find the accelerate executable path.
    accelerate_exec = get_executable_path("accelerate")

    if accelerate_exec == "":
        log.error("Accelerate executable not found.")
        return

    # Configure accelerate launch command.
    run_cmd = [accelerate_exec, "launch", "--config_file", str(path_to_accelerate_config)]
    run_cmd = accelerate_config_cmd(run_cmd=run_cmd)
    run_cmd.append(str(path_to_script))

    # run_cmd.append("--mixed_precision")
    # run_cmd.append("bf16")

    # Add TOML config argument.
    toml_config_path = begin_json_config(str(path_to_flux_config))
    run_cmd.append("--config_file")
    run_cmd.append(toml_config_path)

    # Add extra Flux script arguments.
    run_cmd.append("--output_name")
    run_cmd.append(args.session_name)
    run_cmd.append("--train_data_dir")
    run_cmd.append(temp_train_dir)
    run_cmd.append("--output_dir")
    run_cmd.append(args.output_dir)

    run_cmd.append("--pretrained_model_name_or_path")
    run_cmd.append(str(script_dir.joinpath("models", "flux_base_models", BASE_FLUX_DEV_MODEL_NAME)))
    run_cmd.append("--clip_l")
    run_cmd.append(str(script_dir.joinpath("models", "flux_base_models", BASE_FLUX_DEV_CLIP_NAME)))
    run_cmd.append("--t5xxl")
    run_cmd.append(str(script_dir.joinpath("models", "flux_base_models", BASE_FLUX_DEV_T5_NAME)))
    run_cmd.append("--ae")
    run_cmd.append(str(script_dir.joinpath("models", "flux_base_models", BASE_FLUX_DEV_AE_NAME)))

    # Execute the command.
    executed_subprocess = execute_cmd(run_cmd=run_cmd)

    # Check to see if it has finished training.
    is_finished_training(process=executed_subprocess)

    # Once training has finished, ensure that subprocesses are killed.
    terminate_subprocesses(process=executed_subprocess)

    return


if __name__ == "__main__":
    # Start training script.

    # Set up the parser.
    parser_train = setup_parser()
    train_args = parser_train.parse_args()

    log.info("Beginning Flux.1 [dev] Lora training.")

    # Check if the base models are in the correct directory.
    model_status = are_models_verified_flux(log=log)

    if not model_status:
        log.info("Exiting due to a complication.")
        sys.exit()

    # Begin training.
    train_flux(args=train_args)

    # Training has compeleted.
    log.info("Training of Flux model using Dreambooth has been completed.")
