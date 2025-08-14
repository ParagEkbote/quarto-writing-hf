from __future__ import annotations

import os
import uuid
from pathlib import Path
from typing import Annotated

import bentoml
import torch
from huggingface_hub import snapshot_download
from PIL import Image

from src.gradio_frontend import demo
from src.monitoring.prometheus import vram_monitor

# Hugging Face token and local cache
hf_token = os.environ.get("HF_TOKEN")
local_dir = snapshot_download(repo_id="black-forest-labs/FLUX.1-dev", token=hf_token)


def save_image(image: Image.Image, output_dir: Path = Path("/tmp")) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{uuid.uuid4().hex}.png"
    image.save(output_path)
    return output_path


@bentoml.service(
    name="flux_lora_service",
    traffic={"timeout": 600},
    resources={"gpu": 1, "gpu_type": "nvidia-l4"},
)
class FluxLoRAService:
    def __init__(self):
        self.pipe = None
        self.current_adapter = None
        self.lora1_triggers = [
            "Cinematic",
            "Photographic",
            "Anime",
            "Manga",
            "Digital art",
            "Pixel art",
            "Fantasy art",
            "Neonpunk",
            "3D Model",
            "Painting",
            "Animation",
            "Illustration",
        ]
        self.lora2_triggers = ["GHIBSKY"]

    @bentoml.on_startup
    def load_model(self):
        from diffusers import DiffusionPipeline
        from diffusers.quantizers import PipelineQuantizationConfig

        self.pipe = DiffusionPipeline.from_pretrained(
            local_dir,  # <- use cached directory
            torch_dtype=torch.bfloat16,
            quantization_config=PipelineQuantizationConfig(
                quant_backend="bitsandbytes_4bit",
                quant_kwargs={
                    "load_in_4bit": True,
                    "bnb_4bit_quant_type": "nf4",
                    "bnb_4bit_compute_dtype": torch.bfloat16,
                },
                components_to_quantize=["transformer"],
            ),
        ).to("cuda")

        self.pipe.enable_lora_hotswap(target_rank=8)
        # Load LoRA adapters as before
        self.pipe.load_lora_weights(
            "data-is-better-together/open-image-preferences-v1-flux-dev-lora",
            weight_name="pytorch_lora_weights.safetensors",
            adapter_name="open-image-preferences",
        )
        self.pipe.load_lora_weights(
            "aleksa-codes/flux-ghibsky-illustration",
            weight_name="lora_v2.safetensors",
            adapter_name="flux-ghibsky",
        )
        self.current_adapter = "open-image-preferences"

    @bentoml.api
    def generate(self, data: dict) -> Annotated[Path, bentoml.validators.ContentType("image/png")]:
        prompt = data["prompt"]
        trigger_word = data.get("trigger_word", "")

        # Switch LoRA adapter if needed
        if trigger_word in self.lora2_triggers and self.current_adapter != "flux-ghibsky":
            self.pipe.set_adapters(["flux-ghibsky"], adapter_weights=[0.8])
            self.current_adapter = "flux-ghibsky"
        elif (
            trigger_word in self.lora1_triggers
            and self.current_adapter != "open-image-preferences"
        ):
            self.pipe.set_adapters(["open-image-preferences"], adapter_weights=[1.0])
            self.current_adapter = "open-image-preferences"

        # Compile components for performance
        self.pipe.text_encoder = torch.compile(
            self.pipe.text_encoder, fullgraph=False, mode="reduce-overhead"
        )
        self.pipe.text_encoder_2 = torch.compile(
            self.pipe.text_encoder_2, fullgraph=False, mode="reduce-overhead"
        )
        self.pipe.vae = torch.compile(self.pipe.vae, fullgraph=False, mode="reduce-overhead")

        with torch.no_grad():
            with vram_monitor(tag="Image Generation", adapter=self.current_adapter, prompt=prompt):
                image = self.pipe(
                    prompt=prompt,
                    height=1024,
                    width=1024,
                    guidance_scale=3.5,
                    num_inference_steps=28,
                    max_sequence_length=512,
                ).images[0]
            return save_image(image)


@bentoml.mount_asgi_app(path="/ui")
def ui(self):
    return demo
