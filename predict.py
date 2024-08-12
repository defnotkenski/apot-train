# Prediction interface for Cog ⚙️
# https://cog.run/python

from cog import BasePredictor, Input, Path
import tempfile
import zipfile
from concept import train_sdxl, setup_parser


class Predictor(BasePredictor):
    def setup(self):
        # Set up goes here

        print("Starting setup.")

    def predict(
            self,
            json_config: Path = Input(default=None, description="JSON Config for training."),
            train_data_zip: Path = Input(default=None, description="Training data in zip format.")
    ) -> Path:
        # Run model training

        # Extract zipped training data contents into a temp directory
        train_data_dir = tempfile.mkdtemp()
        with zipfile.ZipFile(train_data_zip, 'r') as zip_ref:
            zip_ref.extractall(train_data_dir)

        # Create an output directory
        output_dir = Path(tempfile.mkdtemp())
        output = output_dir.joinpath("oberg_dreambooth.safetensors")
        output_zip = output_dir.joinpath("oberg_dreambooth.zip")

        # Set up parser
        parser = setup_parser()
        args = parser.parse_args()

        args.json_config = json_config
        args.train_data_zip = train_data_zip
        args.output_dir = output_dir

        # Run training
        train_sdxl(args=args)

        # Zip the safetensors file
        with zipfile.ZipFile(output, "w") as output_zip:
            output_zip.write("oberg_dreambooth")

        return output
