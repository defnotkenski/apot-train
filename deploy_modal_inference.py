import json
import os
import subprocess
from pathlib import Path
import modal
from typing import Dict
import uuid
from utils import setup_logging

# ====== Constants. ====== #

log = setup_logging()

MOUNT_WORKFLOW_NAME = "apot_v4e_api_test.json"
SCRIPT_DIR = Path.cwd()

# ====== Build functions. ====== #


def download_foundation_models():
    from huggingface_hub import hf_hub_download

    # ====== UNETs. ====== #

    hf_hub_download(
        repo_id="black-forest-labs/FLUX.1-dev",
        filename="flux1-dev.safetensors",
        local_dir=SCRIPT_DIR.joinpath("comfy", "ComfyUI", "models", "unet"),
        token=os.environ["HF_TOKEN"]
    )

    # ====== Encoders. ====== #

    hf_hub_download(
        repo_id="black-forest-labs/FLUX.1-dev",
        filename="ae.safetensors",
        local_dir=SCRIPT_DIR.joinpath("comfy", "ComfyUI", "models", "vae"),
        token=os.environ["HF_TOKEN"]
    )

    # ====== CLIPs. ====== #

    hf_hub_download(
        repo_id="comfyanonymous/flux_text_encoders",
        filename="clip_l.safetensors",
        local_dir=SCRIPT_DIR.joinpath("comfy", "ComfyUI", "models", "clip"),
        token=os.environ["HF_TOKEN"]
    )

    hf_hub_download(
        repo_id="comfyanonymous/flux_text_encoders",
        filename="t5xxl_fp16.safetensors",
        local_dir=SCRIPT_DIR.joinpath("comfy", "ComfyUI", "models", "clip"),
        token=os.environ["HF_TOKEN"]
    )

    # ====== LORAs. ====== #

    hf_hub_download(
        repo_id="Shakker-Labs/FLUX.1-dev-LoRA-add-details",
        filename="FLUX-dev-lora-add_details.safetensors",
        local_dir=SCRIPT_DIR.joinpath("comfy", "ComfyUI", "models", "loras"),
        token=os.environ["HF_TOKEN"]
    )

    # ====== Upscalers. ====== #

    hf_hub_download(
        repo_id="notkenski/apot-upscalers",
        filename="4x_NMKD-Siax_200k.pth",
        local_dir=SCRIPT_DIR.joinpath("comfy", "ComfyUI", "models", "upscale_models"),
        token=os.environ["HF_TOKEN"]
    )

    hf_hub_download(
        repo_id="notkenski/apot-upscalers",
        filename="8x_NMKD-Faces_160000_G.pth",
        local_dir=SCRIPT_DIR.joinpath("comfy", "ComfyUI", "models", "upscale_models"),
        token=os.environ["HF_TOKEN"]
    )

    return

# ====== Modal Images. ====== #


apot_image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("git", "wget")
    .pip_install("comfy-cli==1.1.8", "huggingface_hub")
    .run_commands("comfy --skip-prompt install --nvidia")
    .run_commands(
        "comfy node install ComfyUI_UltimateSDUpscale",
        "comfy node install rgthree-comfy"
    )
    .run_function(
        download_foundation_models, secrets=[modal.Secret.from_name("huggingface-secret")]
    )
)

app = modal.App(name="apot-inference", image=apot_image)

# ====== Modal Functions & Classes. ====== #


@app.function(
    gpu="A10G",
    allow_concurrent_inputs=10,
    concurrency_limit=1,
    container_idle_timeout=10,
)
@modal.web_server(port=8080, startup_timeout=60)
def ui():
    subprocess.Popen("comfy launch -- --listen 0.0.0.0 --port 8080", shell=True)


@app.cls(
    gpu="H100",
    allow_concurrent_inputs=10,
    concurrency_limit=1,
    container_idle_timeout=10,
    secrets=[modal.Secret.from_name("huggingface-secret")],
    mounts=[
        modal.Mount.from_local_file(local_path=f"comfy_workflows/{MOUNT_WORKFLOW_NAME}", remote_path=f"/root/{MOUNT_WORKFLOW_NAME}")
    ]
)
class InferenceClass:

    @modal.enter()
    def setup(self):
        # ====== Launch Comfy in the background. ====== #

        cmd = "comfy launch --background"
        subprocess.run(cmd, shell=True, check=True)

    @modal.method()
    def inference(self, workflow_path: Path):
        # ====== Execute workflow with ComfyUI CLI. ====== #

        cmd = f"comfy run --workflow {str(workflow_path)} --wait --timeout 1200"
        subprocess.run(cmd, shell=True, check=True)

        # ====== Get the file prefix and search output directory for images with said prefix. ====== #

        output_dir = Path.cwd().joinpath("comfy", "ComfyUI", "output")

        workflow = json.loads(workflow_path.read_text())
        file_prefix = [node.get("inputs") for node in workflow.values() if node.get("class_type") == "SaveImage"][0]["filename_prefix"]

        print(f"Looking for files with the prefix: {file_prefix}")

        # ====== DEBUG (ignore). ====== #

        # ls_output = [files for files in os.listdir(output_dir)]
        # print(ls_output)
        # ls_temp = [files for files in os.listdir(Path.cwd().joinpath("comfy", "ComfyUI", "temp"))]
        # print(ls_temp)

        # ====== Once found with prefix, return as bytes. ====== #

        for file in output_dir.iterdir():
            if file.name.startswith(file_prefix):
                return file.read_bytes()

    @modal.web_endpoint(method="POST")
    def api(self, payload: Dict):
        from fastapi import Response
        from huggingface_hub import hf_hub_download

        apot_lora_name = payload["apot_lora_name"]

        # ====== Download custom lora weights. ====== #

        hf_hub_download(
            repo_id="notkenski/flux-loras",
            filename=apot_lora_name,
            local_dir=Path.cwd().joinpath("comfy", "ComfyUI", "models", "loras"),
            token=os.environ["HF_TOKEN"]
        )

        log.info(f"Downloaded {apot_lora_name} from Hugging Face Repo.")

        current_dir = Path.cwd()
        workflow_data = json.loads(current_dir.joinpath(f"{MOUNT_WORKFLOW_NAME}").read_text())

        # ====== Add params to workflow. ====== #

        workflow_data["85"]["inputs"]["lora_1"]["lora"] = apot_lora_name

        workflow_data["6"]["inputs"]["text"] = payload["pos_prompt"]
        workflow_data["51"]["inputs"]["batch_size"] = payload["count"]

        # ====== Give the output image a unique id. ====== #

        client_id = uuid.uuid4().hex
        log.debug(f"The client_id is {client_id}")

        # workflow_data["83"]["inputs"]["filename_prefix"] = client_id
        workflow_data["84"]["inputs"]["filename_prefix"] = client_id

        # ====== Save this workflow to a new file. ====== #

        new_workflow_filename = f"wf_{client_id}.json"
        new_wf_path = Path.cwd().joinpath(new_workflow_filename)

        json.dump(workflow_data, new_wf_path.open("w"))

        # ====== Run inference. ====== #

        img_bytes = self.inference.local(new_wf_path)

        return Response(img_bytes, media_type="image/jpeg")
