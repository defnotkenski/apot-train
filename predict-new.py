# Prediction interface for Cog ⚙️
# https://cog.run/python

from cog import BasePredictor, Input, Path
import tempfile
import zipfile
from sd_scripts.sdxl_train import setup_parser, train


class Predictor(BasePredictor):
    def setup(self) -> None:
        pass

    def predict(
            self,

            # Basic Configurations.

            pretrained_model_name_or_path: Path = Input(default=None, description="Model to be fine-tuned with."),
            output_name: str = Input(default=None, description="Trained model output name."),
            save_model_as: str = Input(default="safetensors", description="Format to save the model in.", choices=[
                "ckpt", "diffusers", "safetensors", "diffusers_safetensors"
            ]),
            save_precision: str = Input(default=None, description="Precision in saving.", choices=[
                "float", "fp16", "bf16"
            ]),
            max_train_epochs: int = Input(default=0, description="Training epochs."),
            max_train_steps: int = Input(default=0, description="Training steps."),
            resolution: str = Input(default="1024,1024", description="Resolution in training"),
            enable_bucket: bool = Input(default=True, description="Enable buckets for multi aspect ratio training."),
            save_every_n_epochs: int = Input(default=None, description="Save checkpoint every N epochs."),
            cache_latents: bool = Input(default=False, description="Cache latents to main memory to reduce VRAM usage."),
            train_data_zip: Path = Input(default=None, description="Directory containing training images."),
            optimizer_type: str = Input(default=None, description="Optimizer to use.", choices=[
                "AdamW", "PagedAdamW", "PagedAdamW8bit", "PagedAdamW32bit", "Lion8bit", "PagedLion8bit", "Lion", "SGDNesterov", "SGDNesterov8bit",
                "DAdaptation(DAdaptAdamPreprint)", "DAdaptAdaGrad", "DAdaptAdam", "DAdaptAdan", "DAdaptAdanIP", "DAdaptLion", "DAdaptSGD",
                "AdaFactor", "Prodigy"
            ]),
            optimizer_args: str = Input(default=None, description="Additional arguments for the optimizer."),
            lr_scheduler: str = Input(default=None, description="Scheduler to use for learning rate.", choices=[
                "linear", "cosine", "cosine_with_restarts", "polynomial", "constant (default)", "constant_with_warmup", "adafactor"
            ]),
            learning_rate: float = Input(default=None, description="Learning rate. If using Prodigy Optimizer, set to 1.0"),
            lr_warmup_steps: int = Input(default=0, description="Number of steps for the warmup in the lr scheduler."),
            learning_rate_te1: float = Input(default=None, description="Learning rate for text encoder 1."),
            learning_rate_te2: float = Input(default=None, description="Learning rate for text encoder 2.")
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
        args.output_dir = output_dir

        args.pretrained_model_name_or_path = pretrained_model_name_or_path
        args.output_name = output_name
        args.save_model_as = save_model_as
        args.save_precision = save_precision
        args.max_train_epochs = max_train_epochs
        args.max_train_steps = max_train_steps
        args.resolution = resolution
        args.enable_bucket = enable_bucket
        args.save_every_n_epochs = save_every_n_epochs
        args.cache_latents = cache_latents
        args.train_data_dir = train_data_dir
        args.optimizer_type = optimizer_type
        args.optimizer_args = optimizer_args
        args.lr_scheduler = lr_scheduler
        args.learning_rate = learning_rate
        args.lr_warmup_steps = lr_warmup_steps
        args.learning_rate_te1 = learning_rate_te1
        args.learning_rate_te2 = learning_rate_te2

        return output_dir
