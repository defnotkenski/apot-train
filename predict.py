import sys
import time
from cog import BasePredictor, Input, Path as cogPath
from pathlib import Path
import tempfile
import zipfile
from main_flux import train_flux
import torch
from subprocess import check_call
from argparse import Namespace
import gc
from utils import setup_logging, are_models_verified_flux

log = setup_logging()


class Predictor(BasePredictor):
    def setup(self):
        # Wait a little bit for instance to be ready and run set up.
        log.info("Starting setup.")

        time.sleep(10)
        check_call("nvidia-smi", shell=True)
        assert torch.cuda.is_available()

    def predict(
            self,
            session_name: str = Input(default=None, description="Name of training session."),
            train_zip: cogPath = Input(default=None, description="Training data in zip format.")
    ) -> cogPath:
        # Run model training and clear GPU memory.
        log.info("Starting training...")

        torch.cuda.empty_cache()
        gc.collect()

        # Check to make sure correct models are already in the appropriate dir.
        if not are_models_verified_flux(log=log):
            sys.exit()

        # Create temporary output directory to store output.
        log.info("Creating the temp output directory.")
        temp_output_dir = tempfile.mkdtemp()

        # Create paths to the output safetensors file and zip by appending output directory.
        log.info("Creating path to safetensors and zipfile.")
        path_output_safetensors = Path(temp_output_dir).joinpath(f"{session_name}.safetensors")
        path_output_zip = Path(temp_output_dir).joinpath(f"{session_name}.zip")

        # Assign args to Namespace in order to pass to the imported training function.
        args = {
            "session_name": session_name,
            "training_dir": str(train_zip),
            "output_dir": temp_output_dir,
        }

        args_namespace = Namespace(**args)

        # Run training script.
        log.info("Running training script.")
        train_flux(args=args_namespace)

        # Check to see if safetensor output exists.
        log.info("Validating safetensors output.")

        if not path_output_safetensors.exists():
            log.error("Safetensors output does not exist in the output directory. Exiting...")
            sys.exit()

        # Clean shit up and free up resources.
        log.info("Cleaning shit up.")
        gc.collect()
        torch.cuda.empty_cache()

        # Zip the safetensors file and return to Replicate.
        log.info("Zipping safetensors file.")
        with zipfile.ZipFile(path_output_zip, "w") as zip_write:
            zip_write.write(path_output_safetensors)

        # Check to see if zip file exists now.
        log.info("Validating ZIP files exists.")

        if not path_output_zip.exists():
            log.error("The output zip does not exist in the output dir. Exiting...")
            sys.exit()

        log.info(f"Training of {session_name} has been completed.")

        return path_output_zip
