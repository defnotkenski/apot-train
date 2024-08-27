import argparse
import sys
import zipfile
import torch.cuda
import gc
from utils import *
from huggingface_hub import HfApi

# Some variable setups to be commonly used throughout this script. Varibles in UPPERCASE are subject to change by the user.
log = setup_logging()
script_dir = Path.cwd()
python = sys.executable

REPLICATE_REPO_ID = "notkenski/apothecary-dev"

# Add sd_scripts_flux submodule to python's path.
sys.path.insert(0, str(script_dir.joinpath("sd_scripts")))


def setup_parser() -> argparse.ArgumentParser:
    # Set up the parser to accept CLI arguments.

    parser = argparse.ArgumentParser()
    parser.add_argument("--session_name", default=None, required=True, help="Name of this training session (Will appear as file names).")
    parser.add_argument("--training_dir", default=None, required=True, help="Path of training data in zip format.")
    parser.add_argument("--output_dir", default=None, required=True, help="Path to the local output directory.")
    parser.add_argument("--upload", default=None, required=False, help="Whether or not to upload to Huggingface Repo using token.")

    # Automatically set, but can be user-defined in the CLI.
    # parser.add_argument("--flux_config", default=None, required=True, help="Configuration JSON file for Flux training.")
    # parser.add_argument("--clip_l", default=None, required=True, help="Path to the clip_l model.")
    # parser.add_argument("--t5xxl", default=None, required=True, help="Path to the t5xxl model.")
    # parser.add_argument("--ae", default=None, required=True, help="Path to the ae model.")

    return parser


def train_flux(args: argparse.Namespace) -> None:
    # Begin training of the Flux model.

    # Create appropriate paths to files.
    path_to_script = script_dir.joinpath("sd_scripts", "flux_train_network.py")
    path_to_accelerate_config = script_dir.joinpath("configs", "accelerate.yaml")
    path_to_flux_config = script_dir.joinpath("configs", "flux_lora.json")

    # Unzip file and store in temp directory.
    temp_train_dir = tempfile.mkdtemp()
    with zipfile.ZipFile(args.training_dir, "r") as zip_ref:
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

    # Add TOML config argument.
    toml_config_path = begin_json_config(str(path_to_flux_config))
    run_cmd.append("--config_file")
    run_cmd.append(toml_config_path)

    # Add extra Flux script arguments.
    run_cmd.append("--output_name")
    run_cmd.append(f"{args.session_name}_lora")
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
    executed_subprocess = execute_cmd(run_cmd=run_cmd, log=log)

    # Check to see if it has finished training.
    is_finished_training(process=executed_subprocess, log=log)

    # Once training has finished, ensure that subprocesses are killed.
    terminate_subprocesses(process=executed_subprocess, log=log)

    return


if __name__ == "__main__":
    # Set up the parser.
    parser_train = setup_parser()
    train_args = parser_train.parse_args()

    # Start training script.
    log.info("Beginning Flux.1 [dev] training.")

    # Clear GPU memory.
    log.info("Clearing GPU memory for training.")

    torch.cuda.empty_cache()
    gc.collect()

    # Check if the base models are in the correct directory.
    model_status = are_models_verified_flux(log=log)

    if not model_status:
        sys.exit()

    # Begin training.
    train_flux(args=train_args)

    # Upload to Huggingface Repository.
    try:
        if train_args.upload is not None:
            log.info("[reverse cyan1]Starting upload to Huggingface Hub.", extra={"markup": True})

            hf_api = HfApi()
            upload_output_path = Path(train_args.output_dir).joinpath(f"{train_args.session_name}_lora.safetensors")
            hf_api.upload_file(
                token=train_args.upload,
                path_or_fileobj=upload_output_path,
                path_in_repo=upload_output_path.name,
                repo_id=REPLICATE_REPO_ID
            )
    except Exception as e:
        log.error(f"Exception during Huggingface upload: {e}")

    # Training has compeleted.
    log.info("Training of Flux model has been completed.")
