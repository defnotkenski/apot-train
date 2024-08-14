# Prediction interface for Cog ⚙️
# https://cog.run/python

from cog import BasePredictor, Input, Path
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
            json_config: Path = Input(default=None, description="JSON Config for training."),
            train_data_zip: Path = Input(default=None, description="Training data in zip format.")
    ) -> Path:
        # Run model training

        try:
            # Extract zipped training data contents into a temp directory
            train_data_dir = tempfile.mkdtemp()
            with zipfile.ZipFile(train_data_zip, 'r') as zip_ref:
                zip_ref.extractall(train_data_dir)

            # Create output directories
            output_dir = tempfile.mkdtemp()
            output_tensors = Path(output_dir).joinpath("oberg_dreambooth.safetensors")
            output_zip = Path(output_dir).joinpath("oberg_dreambooth.zip")

            # Set up parser
            parser = setup_parser()
            args = parser.parse_args()

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
