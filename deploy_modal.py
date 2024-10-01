import logging
import os
import sys
import tempfile
from pathlib import Path
import modal
from huggingface_hub import hf_hub_download
from main_flux import train_flux
import subprocess
from argparse import Namespace
from utils import upload_to_huggingface, setup_logging

app = modal.App("apot-train-flux")

apot_image = (
    modal.Image.debian_slim(python_version="3.10")
    .apt_install("libglib2.0-0", "libsm6", "libxrender1", "libxext6", "ffmpeg", "libgl1", "git")
    .run_commands(
        "pip install torch==2.4.0 --index-url https://download.pytorch.org/whl/cu124",
        "pip install torchvision==0.19.0 --index-url https://download.pytorch.org/whl/cu124"
    )
    .pip_install(
        "accelerate==0.33.0",
        "transformers==4.44.0",
        "diffusers[torch]==0.25.0",
        "ftfy==6.1.1",
        "opencv-python==4.7.0.68",
        "einops==0.7.0",
        "pytorch-lightning==1.9.0",
        "bitsandbytes==0.43.3",
        "prodigyopt==1.0",
        "lion-pytorch==0.0.6",
        "tensorboard",
        "safetensors==0.4.4",
        "altair==4.2.2",
        "easygui==0.98.3",
        "toml==0.10.2",
        "voluptuous==0.13.1",
        "huggingface-hub==0.24.5",
        "imagesize==1.4.1",
        "rich==13.*",
        "sentencepiece==0.2.0",
        "psutil==6.*",
        "pyyaml==6.*",
        "numpy==1.*",
        "slack_sdk==3.*"
    )
    .run_commands(
        "cd root/ && git init .",
        "cd root/ && git submodule add -b sd3 https://github.com/kohya-ss/sd-scripts.git sd_scripts"
    )
)


@app.cls(
    image=apot_image,
    timeout=7200,
    gpu=["H100"],
    secrets=[
        modal.Secret.from_name("huggingface-secret")
    ],
    mounts=[
        modal.Mount.from_local_dir(Path.cwd().joinpath("configs"), remote_path="/root/configs", recursive=True),
        modal.Mount.from_local_dir(Path.cwd().joinpath("models"), remote_path="/root/models", recursive=True)
    ])
class ApotTrain:
    temp_output_dir: Path = Path(tempfile.mkdtemp())
    temp_input_dir: Path = Path(tempfile.mkdtemp())
    log: logging.Logger = setup_logging()

    @modal.build()
    def download_model_weights(self):
        print("Downloading model weights!")

        weights_folder = Path.cwd().joinpath("models", "flux_base_models")
        os.makedirs(weights_folder, exist_ok=True)

        # hf_hub_download(repo_id="black-forest-labs/FLUX.1-dev", filename="flux1-dev.safetensors", local_dir=weights_folder)
        # hf_hub_download(repo_id="black-forest-labs/FLUX.1-dev", filename="ae.safetensors", local_dir=weights_folder)

        # hf_hub_download(repo_id="comfyanonymous/flux_text_encoders", filename="t5xxl_fp16.safetensors", local_dir=weights_folder)
        # hf_hub_download(repo_id="comfyanonymous/flux_text_encoders", filename="clip_l.safetensors", local_dir=weights_folder)

        return

    @staticmethod
    def are_weights_verified() -> bool:
        base_path = Path.cwd().joinpath("models", "flux_base_models")
        weights_flux_dev = base_path.joinpath("flux1-dev.safetensors")
        weights_t5_xxl = base_path.joinpath("t5xxl_fp16.safetensors")
        weights_clip_l = base_path.joinpath("clip_l.safetensors")
        weights_ae = base_path.joinpath("ae.safetensors")

        if weights_flux_dev.exists() and weights_ae.exists() and weights_t5_xxl.exists() and weights_clip_l.exists():
            return True

        return False

    @modal.method()
    def train(self, session_name: str, training_images_name: str):
        print("Training model!")

        print(f"The current working directory is: {str(Path.cwd())}")
        subprocess.check_call("ls -l", shell=True)
        subprocess.check_call("ls -l models/flux_base_models", shell=True)

        self.log.info("Downloading training images...")
        hf_hub_download(repo_id="notkenski/apothecary-dev", filename=training_images_name, local_dir=self.temp_input_dir)

        if not self.are_weights_verified():
            print("Weights are not verified!")
            sys.exit()

        print("Weights are verified.")

        args = {
            "session_name": session_name,
            "training_dir": str(self.temp_input_dir.joinpath(training_images_name)),
            "output_dir": str(self.temp_output_dir),
            "upload": os.environ["HF_TOKEN"]
        }

        train_namespace = Namespace(**args)

        train_flux(args=train_namespace)

        # ====== Upload to Huggingface ====== #

        path_model = self.temp_output_dir.joinpath(f"{session_name}.safetensors")
        path_model_yaml = Path.cwd().joinpath("configs", "flux_lora.yaml")
        upload_to_huggingface(model_path=path_model, yaml_path=path_model_yaml, base_path="flux/loras", train_args=train_namespace, log=self.log)

        return


@app.local_entrypoint()
def main():
    apot = ApotTrain()
    result = apot.train.remote(session_name="modal_pokimane_3600", training_images_name="datasets/pokimane_3_sw1ft_woman.zip")

    print(result)
