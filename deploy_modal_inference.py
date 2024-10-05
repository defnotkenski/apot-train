import json
import os
import subprocess
from pathlib import Path
import modal
from typing import Dict
import uuid

# ====== Modal Images. ====== #

app = modal.App(name="apot-inference")

apot_image = (
    modal.Image.debian_slim(python_version="3.10", force_build=False)
    .apt_install("git", "wget")
    .pip_install("comfy-cli==1.1.8")
    .run_commands("comfy --skip-prompt install --nvidia")
    .run_commands(
        "comfy node install ComfyUI_UltimateSDUpscale",
        "comfy node install rgthree-comfy",
        "comfy node install ComfyUI-SUPIR"
    )
    .run_commands(
        "wget --content-disposition --no-verbose 'https://huggingface.co/Shakker-Labs/FLUX.1-dev-LoRA-add-details/resolve/main/FLUX-dev-lora-add_details.safetensors?download=true' -P /root/comfy/ComfyUI/models/loras",
        "wget --content-disposition --no-verbose 'https://huggingface.co/notkenski/apothecary-dev/resolve/main/flux/loras/modal_pokimane_3600/modal_pokimane_3600.safetensors?download=true' -P /root/comfy/ComfyUI/models/loras"
    )
    # .copy_local_file(local_path="models/flux_base_models/flux1-dev.safetensors", remote_path="root/comfy/comfyui/models/unet")
)

# ====== Modal Functions & Classes. ====== #


@app.function(
    image=apot_image,
    gpu="A100",
    allow_concurrent_inputs=10,
    concurrency_limit=1,
    mounts=[
        modal.Mount.from_local_file(local_path="models/flux_base_models/flux1-dev.safetensors", remote_path="root/comfy/ComfyUI/models/unet/flux1-dev.safetensors"),
        modal.Mount.from_local_file(local_path="models/flux_base_models/ae.safetensors", remote_path="root/comfy/ComfyUI/models/vae/ae.safetensors"),
        modal.Mount.from_local_file(local_path="models/flux_base_models/clip_l.safetensors", remote_path="root/comfy/ComfyUI/models/clip/clip_l.safetensors"),
        modal.Mount.from_local_file(local_path="models/flux_base_models/t5xxl_fp16.safetensors", remote_path="root/comfy/ComfyUI/models/clip/t5xxl_fp16.safetensors")
    ]
)
@modal.web_server(port=8080, startup_timeout=60)
def ui():
    subprocess.Popen("comfy launch -- --listen 0.0.0.0 --port 8080", shell=True)


@app.cls(
    image=apot_image,
    gpu=modal.gpu.A100(size="80GB"),
    allow_concurrent_inputs=10,
    # container_idle_timeout=300,
    mounts=[
        modal.Mount.from_local_file(local_path=Path.cwd().joinpath("models", "flux_base_models", "flux1-dev.safetensors"), remote_path="/root/comfy/ComfyUI/models/unet/flux1-dev.safetensors"),
        modal.Mount.from_local_file(local_path="models/flux_base_models/ae.safetensors", remote_path="/root/comfy/ComfyUI/models/vae/ae.safetensors"),
        modal.Mount.from_local_file(local_path="models/flux_base_models/clip_l.safetensors", remote_path="/root/comfy/ComfyUI/models/clip/clip_l.safetensors"),
        modal.Mount.from_local_file(local_path="models/flux_base_models/t5xxl_fp16.safetensors", remote_path="/root/comfy/ComfyUI/models/clip/t5xxl_fp16.safetensors"),
        modal.Mount.from_local_file(local_path="comfy_workflows/flux_v4e_api.json", remote_path="/root/flux_v4e_api.json")
    ]
)
class InferenceClass:
    @modal.enter()
    def launch_comfy_background(self):
        cmd = "comfy launch --background"
        subprocess.run(cmd, shell=True, check=True)

    @modal.method()
    def inference(self, workflow_path: Path):
        cmd = f"comfy run --workflow {str(workflow_path)} --wait --timeout 1200"
        subprocess.run(cmd, shell=True, check=True)

        output_dir = "root/comfy/ComfyUI/output"

        workflow = json.loads(Path(workflow_path).read_text())
        save_image_node = [node.get("inputs") for node in workflow.values() if node.get("class_type") == "SaveImage"]
        file_prefix = save_image_node[0]["filename_prefix"]

        for file in Path(output_dir).iterdir():
            if file.name.startswith(file_prefix):
                return file.read_bytes()

    @modal.web_endpoint(method="POST")
    def api(self, payload: Dict):
        from fastapi import Response

        current_dir = Path.cwd()
        workflow_data = json.loads(current_dir.joinpath("flux_v4e_api.json").read_text())

        # ====== Add params to workflow. ====== #

        workflow_data["6"]["inputs"]["text"] = payload["pos_prompt"]
        # workflow_data["51"]["inputs"]["batch_size"] = payload["count"]
        workflow_data["66"]["inputs"]["lora_1"]["lora"] = "modal_pokimane_3600.safetensors"

        # ====== Give the output image a unique id. ====== #

        client_id = uuid.uuid4().hex
        workflow_data["83"]["inputs"]["filename_prefix"] = client_id

        # ====== Save this workflow to a new file. ====== #

        new_workflow_filename = f"{client_id}_api.json"
        new_wf_path = Path.cwd().joinpath(new_workflow_filename)
        json.dump(workflow_data, new_wf_path.open("w"))

        # ====== Run inference. ====== #

        img_bytes = self.inference.local(Path.cwd().joinpath("flux_v4e_api.json"))

        return Response(img_bytes, media_type="image/jpeg")
