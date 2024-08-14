# Prediction interface for Cog ⚙️
# https://cog.run/python

from cog import BasePredictor, Input, Path as cogPath
from pathlib import Path
import tempfile
import zipfile
from concept import train_sdxl, setup_parser
import mimetypes


class Predictor(BasePredictor):
    def setup(self):
        # Set up goes here

        print("Starting setup.")
        mimetypes.add_type("application/octet-stream", ".safetensors")

    def predict(
            self,
            json_config: cogPath = Input(default=None, description="JSON Config for training."),
            train_data_zip: cogPath = Input(default=None, description="Training data in zip format.")
    ) -> cogPath:
        # Run model training

        print("Starting model training nigga.")
        try:
            # Extract zipped training data contents into a temp directory
            print("Extracting zip file contents into a temp dir.")
            train_data_dir = tempfile.mkdtemp()
            with zipfile.ZipFile(train_data_zip, 'r') as zip_ref:
                zip_ref.extractall(train_data_dir)

            # Create output directories
            print("Creating the output dirs.")
            output_dir = tempfile.mkdtemp()
            print("1")
            output_tensors = Path(output_dir).joinpath("oberg_dreambooth.safetensors")
            print("2")
            output_zip = Path(output_dir).joinpath("oberg_dreambooth.zip")
            print("3")

            # Set up parser
            print("Setting up the parsers.")
            parser = setup_parser()
            print("Parsing args.")
            args = parser.parse_args()

            print("Assigning arguments to parsers.")
            args.json_config = json_config
            args.train_data_zip = train_data_zip
            args.output_dir = output_dir

            # Run training
            train_sdxl(args=args)

            # Zip the safetensors file
            with zipfile.ZipFile(output_zip, "w") as zip_write:
                zip_write.write(output_tensors)

            return output_zip

        except Exception as e:
            print(f"Error in predict function: {e}")
