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


class Predictor(BasePredictor):
    def setup(self):
        # Wait a little bit for instance to be ready and run set up

        print("Starting setup.")

        time.sleep(10)
        check_call("nvidia-smi", shell=True)
        assert torch.cuda.is_available()

    def predict(
            self,
            json_config: cogPath = Input(default=None, description="JSON Config for training."),
            train_data_zip: cogPath = Input(default=None, description="Training data in zip format.")
    ) -> cogPath:
        # Run model training

        print("Starting model training nigga.")

        # Extract zipped training data contents into a temp directory
        print("Extracting zip file contents into a temp dir.")
        train_data_dir = tempfile.mkdtemp()
        with zipfile.ZipFile(train_data_zip, 'r') as zip_ref:
            zip_ref.extractall(train_data_dir)

        # Create output directories
        print("Creating the output dirs.")
        output_dir = tempfile.mkdtemp()

        # Log system usages
        print(f"RAM USAGE: {psutil.virtual_memory().percent}")
        print(check_call("nvidia-smi", shell=True))

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

        print("Assigning args to Namespace.")
        args = {
            "json_config": json_config,
            "train_data_zip": train_data_zip,
            "output_dir": output_dir
        }

        args = Namespace(**args)

        # Run training
        try:
            print("Running training.")
            train_sdxl(args=args)
        except Exception as e:
            print(f"An exception occured when running training script: {e}")

        print("Adding path to safetensors.")
        output_tensors = Path(output_dir).joinpath("oberg_dreambooth.safetensors")
        print("Adding path to zip file.")
        output_zip = Path(output_dir).joinpath("oberg_dreambooth.zip")

        # Zip the safetensors file
        print("Zipping safetensors file.")
        with zipfile.ZipFile(output_zip, "w") as zip_write:
            zip_write.write(output_tensors)

        return output_zip
