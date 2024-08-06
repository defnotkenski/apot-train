# Prediction interface for Cog ⚙️
# https://cog.run/python

from cog import BasePredictor, Input, Path
import tempfile
import zipfile
from sd_scripts.sdxl_train import setup_parser, train
from sd_scripts.library.config_util import load_user_config


class Predictor(BasePredictor):
    def setup(self) -> None:
        pass

    def predict(
            self,

            # Basic Configurations.
            dataset_config: Path = Input(default=None, description="Config file for detail settings."),

            train_data_zip: Path = Input(default=None, description="Directory containing training images."),
            pretrained_model_name_or_path: Path = Input(default=None, description="Model to be fine-tuned with."),
            output_name: str = Input(default=None, description="Base name of trained model file.")
    ) -> Path:
        # Extract zipped training data contents into a temp directory.
        train_data_dir = tempfile.mkdtemp()
        with zipfile.ZipFile(train_data_zip, 'r') as zip_ref:
            zip_ref.extractall(train_data_dir)

        # Create an output directory.
        output_dir = Path(tempfile.mkdtemp())

        # Set up parser for SD_Scripts
        parser = setup_parser()
        args = parser.parse_args()

        # Assign arguments.
        args.dataset_config = dataset_config

        args.output_dir = output_dir
        args.train_data_dir = train_data_dir
        args.pretrained_model_name_or_path = pretrained_model_name_or_path
        args.output_name = output_name

        return output_dir
