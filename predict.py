# Prediction interface for Cog ⚙️
# https://cog.run/python
import time

from cog import BasePredictor, Input, Path as cogPath
from pathlib import Path
import tempfile
import zipfile
from concept import train_sdxl
import torch
from subprocess import check_call
from argparse import Namespace
import psutil
import gc
from utils import setup_logging

log = setup_logging()


class Predictor(BasePredictor):
    def setup(self):
        # Wait a little bit for instance to be ready and run set up

        log.info("Starting setup.")

        time.sleep(10)
        check_call("nvidia-smi", shell=True)
        assert torch.cuda.is_available()

    def predict(
            self,
            json_config: cogPath = Input(default=None, description="JSON Config for training."),
            train_data_zip: cogPath = Input(default=None, description="Training data in zip format.")
    ) -> cogPath:
        # Run model training

        log.info("Starting model training nigga.")

        # Extract zipped training data contents into a temp directory
        log.info("Extracting zip file contents into a temp dir.")
        train_data_dir = tempfile.mkdtemp()
        with zipfile.ZipFile(train_data_zip, 'r') as zip_ref:
            zip_ref.extractall(train_data_dir)

        # Create output directories
        log.info("Creating the output dirs.")
        output_dir = tempfile.mkdtemp()

        # Log system usages
        log.info(f"RAM USAGE: {psutil.virtual_memory().percent}")
        log.info(check_call("nvidia-smi", shell=True))

        # Set up parser
        # print("Setting up the parsers.")
        # parser = setup_parser()
        # print("Parsing args.")
        # args = parser.parse_args()
        # print("Done parsing args.")

        # print("Assigning arguments to parsers.")
        # args.json_config = json_config
        # args.train_data_zip = train_data_zip
        # args.output_dir = output_dir

        log.info("Assigning args to Namespace.")
        args = {
            "json_config": json_config,
            "train_data_zip": train_data_zip,
            "output_dir": output_dir
        }

        args = Namespace(**args)

        # Run training
        try:
            log.info("Running training.")
            train_sdxl(args=args)
        except Exception as e:
            log.info(f"An exception occured when running training script: {e}")

        # Clean shit up
        gc.collect()
        torch.cuda.empty_cache()

        log.info("Adding path to safetensors.")
        output_tensors = Path(output_dir).joinpath("oberg_dreambooth.safetensors")
        log.info("Adding path to zip file.")
        output_zip = Path(output_dir).joinpath("oberg_dreambooth.zip")

        # Zip the safetensors file
        log.info("Zipping safetensors file.")
        with zipfile.ZipFile(output_zip, "w") as zip_write:
            zip_write.write(output_tensors)

        return output_zip
