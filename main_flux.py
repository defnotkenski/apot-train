import argparse
import sys
import zipfile
import torch.cuda
import gc
import tempfile
from pathlib import Path
from huggingface_hub import HfApi
import yaml
from utils import (
    setup_logging, get_executable_path, accelerate_config_cmd, convert_to_toml_config, execute_cmd, is_finished_training,
    terminate_subprocesses, are_models_verified_flux, BASE_FLUX_DEV_MODEL_NAME, BASE_FLUX_DEV_CLIP_NAME, BASE_FLUX_DEV_T5_NAME,
    BASE_FLUX_DEV_AE_NAME
)

# Some variable setups to be commonly used throughout this script. Varibles in UPPERCASE are subject to change by the user.
log = setup_logging()
script_dir = Path.cwd()
python = sys.executable
temp_output_dir = Path(tempfile.mkdtemp(prefix="output_"))

path_to_accelerate_config = script_dir.joinpath("configs", "accelerate.yaml")
path_to_accelerate_exec = get_executable_path("accelerate")

REPLICATE_REPO_ID = "notkenski/apothecary-dev"

# Add sd_scripts_flux submodule to python's path.
sys.path.insert(0, str(script_dir.joinpath("sd_scripts")))


def setup_parser() -> argparse.ArgumentParser:
    # Set up the parser to accept CLI arguments.

    parser = argparse.ArgumentParser()
    parser.add_argument("--session_name", default=None, required=True, help="Name of this training session (Will appear as file names).")
    parser.add_argument("--training_dir", default=None, required=True, help="Path of training data in zip format.")
    parser.add_argument("--upload", default=None, required=False, help="Whether or not to upload to Huggingface Repo using token.")
    parser.add_argument("--output_dir", default=None, required=True, help="Path to the local output directory.")

    # Automatically set, but can be user-defined in the CLI.
    # parser.add_argument("--flux_config", default=None, required=True, help="Configuration JSON file for Flux training.")
    # parser.add_argument("--clip_l", default=None, required=True, help="Path to the clip_l model.")
    # parser.add_argument("--t5xxl", default=None, required=True, help="Path to the t5xxl model.")
    # parser.add_argument("--ae", default=None, required=True, help="Path to the ae model.")

    return parser


def train_flux(args: argparse.Namespace) -> None:
    # Begin training of the Flux model.

    # Create appropriate paths to files.
    path_to_script = script_dir.joinpath("sd_scripts", "flux_train.py")
    path_to_flux_config = script_dir.joinpath("configs", "flux_dreambooth.yaml")

    # Unzip file and store in temp directory.
    temp_train_dir = tempfile.mkdtemp()
    with zipfile.ZipFile(args.training_dir, "r") as zip_ref:
        zip_ref.extractall(temp_train_dir)

    # Configure accelerate launch command.
    run_cmd = [path_to_accelerate_exec, "launch", "--config_file", str(path_to_accelerate_config)]
    run_cmd = accelerate_config_cmd(run_cmd=run_cmd)
    run_cmd.append(str(path_to_script))

    # Convert the YAML file into a TOML config argument.
    toml_config_path = convert_to_toml_config(str(path_to_flux_config))
    run_cmd.append("--config_file")
    run_cmd.append(toml_config_path)

    # Add extra Flux script arguments.
    run_cmd.append("--output_name")
    run_cmd.append(f"{args.session_name}_dreambooth")
    run_cmd.append("--train_data_dir")
    run_cmd.append(temp_train_dir)
    run_cmd.append("--output_dir")
    run_cmd.append(str(temp_output_dir))

    run_cmd.append("--pretrained_model_name_or_path")
    run_cmd.append(str(script_dir.joinpath("models", "flux_base_models", BASE_FLUX_DEV_MODEL_NAME)))
    run_cmd.append("--clip_l")
    run_cmd.append(str(script_dir.joinpath("models", "flux_base_models", BASE_FLUX_DEV_CLIP_NAME)))
    run_cmd.append("--t5xxl")
    run_cmd.append(str(script_dir.joinpath("models", "flux_base_models", BASE_FLUX_DEV_T5_NAME)))
    run_cmd.append("--ae")
    run_cmd.append(str(script_dir.joinpath("models", "flux_base_models", BASE_FLUX_DEV_AE_NAME)))

    # Execute the command.
    executed_subprocess = execute_cmd(run_cmd=run_cmd, log=log)

    # Check to see if it has finished training.
    is_finished_training(process=executed_subprocess, log=log)

    # Once training has finished, ensure that subprocesses are killed.
    terminate_subprocesses(process=executed_subprocess, log=log)

    return


def extract_flux_lora(args: argparse.Namespace) -> None:
    # Extract the lora from the fine-tuned Flux model.

    # There is no config file argument for the lora extraction script, so I need to manually place them.
    with open("configs/flux_xlora.yaml", "r") as read_xlora_config:
        raw_xlora_config = yaml.safe_load(read_xlora_config)

    # Clean up the config file to only contain arguments with a value.
    xlora_config = {
        key: raw_xlora_config[key] for key in raw_xlora_config if raw_xlora_config[key] not in [""]
    }

    # Creat paths to the appropriate files.
    path_to_script = script_dir.joinpath("sd_scripts", "networks", "flux_extract_lora.py")
    path_to_original_model = script_dir.joinpath("models", "flux_base_models", "flux1-dev.safetensors")
    path_to_finetuned_model = temp_output_dir.joinpath(f"{args.session_name}_dreambooth.safetensors")
    path_to_save = temp_output_dir.joinpath(f"{args.session_name}_xlora.safetensors")

    # Formulate the run command.
    run_cmd = [path_to_accelerate_exec, "launch", "--config_file", str(path_to_accelerate_config)]
    run_cmd = accelerate_config_cmd(run_cmd=run_cmd)

    run_cmd.append(str(path_to_script))
    run_cmd.append("--model_org")
    run_cmd.append(str(path_to_original_model))
    run_cmd.append("--model_tuned")
    run_cmd.append(str(path_to_finetuned_model))
    run_cmd.append("--save_to")
    run_cmd.append(str(path_to_save))

    for key in xlora_config:
        if key not in ["model_org", "model_tuned", "save_to"]:
            run_cmd.append(f"--{key}")

            if xlora_config[key] is not True:
                run_cmd.append(str(xlora_config[key]))

    # Execute the command.
    xlora_subprocess = execute_cmd(run_cmd=run_cmd, log=log)

    # Check to see if it has finished.
    is_finished_training(xlora_subprocess, log=log)

    # Make sure the subprocesses are all terminated at finish.
    terminate_subprocesses(xlora_subprocess, log=log)


if __name__ == "__main__":
    # Set up the parser.
    parser_train = setup_parser()
    train_args = parser_train.parse_args()

    # Start training script.
    log.info("[reverse wheat1]Beginning Flux-Dev training.", extra={"markup": True})

    # Clear GPU memory.
    log.info("Clearing GPU memory for training.")

    torch.cuda.empty_cache()
    gc.collect()

    # Check if the base models are in the correct directory.
    model_status = are_models_verified_flux(log=log)

    if not model_status:
        sys.exit()

    # Begin training the Flux.1 [Dev] model.
    train_flux(args=train_args)

    # Extract the lora from the fine-tuned model.
    log.info("[reverse wheat1]Beginning Flux-Dev Lora extraction.", extra={"markup": True})
    extract_flux_lora(args=train_args)

    # Upload to Huggingface Repository.
    try:
        if train_args.upload is not None:
            log.info("[reverse wheat1]Starting upload to Huggingface Hub.", extra={"markup": True})

            hf_api = HfApi()
            upload_output_path = temp_output_dir.joinpath(f"{train_args.session_name}_xlora.safetensors")
            hf_api.upload_file(
                token=train_args.upload,
                path_or_fileobj=upload_output_path,
                path_in_repo=upload_output_path.name,
                repo_id=REPLICATE_REPO_ID
            )
    except Exception as e:
        log.error(f"Exception during Huggingface upload: {e}")

    # Training has compeleted.
    log.info("[reverse honeydew2]Training of Flux-Dev model has been completed.", extra={"markup": True})
