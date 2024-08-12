# Prediction interface for Cog ⚙️
# https://cog.run/python

from cog import BasePredictor, Input, Path
import tempfile
import zipfile
from concept import train_sdxl


class Predictor(BasePredictor):
    def setup(self):
        # Don't really know what to put here.

        print("Starting setup.")

    def predict(
            self,
            json_config: Path = Input(default=None, description="JSON Config for training."),
            train_data_zip: Path = Input(default=None, description="Training data in zip format.")
    ) -> Path:
        # Run model inference

        # Extract zipped training data contents into a temp directory.
        train_data_dir = tempfile.mkdtemp()
        with zipfile.ZipFile(train_data_zip, 'r') as zip_ref:
            zip_ref.extractall(train_data_dir)

        # Create an output directory.
        output_dir = Path(tempfile.mkdtemp())

        return output_dir
