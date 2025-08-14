from __future__ import annotations

import os
import uuid
import time
import psutil
from pathlib import Path
from contextlib import contextmanager
from typing import Annotated
from huggingface_hub import hf_hub_download

import bentoml
from PIL import Image  # Import early so type hints work

hf_token = os.environ.get("HF_TOKEN")

from huggingface_hub import snapshot_download

local_dir = snapshot_download(
    repo_id="black-forest-labs/FLUX.1-dev",
    token=hf_token
)

local_dir = snapshot_download(
    repo_id="black-forest-labs/FLUX.1-dev",
    token=hf_token
)

@bentoml.service(
    name="flux_lora_service",
    traffic={"timeout": 600},
    resources={"gpu": 1, "gpu_type": "nvidia-l4"}
)
class FluxLoRAService:
    def __init__(self):
        self.pipe = None
        self.current_adapter = None
        self.lora1_triggers = [
            "Cinematic", "Photographic", "Anime", "Manga", "Digital art",
            "Pixel art", "Fantasy art", "Neonpunk", "3D Model",
            "Painting", "Animation", "Illustration",
        ]
        self.lora2_triggers = ["GHIBSKY"]

    @bentoml.on_startup
    def load_model(self):
        import torch
        from diffusers import DiffusionPipeline
        from diffusers.quantizers import PipelineQuantizationConfig

        # Load pipeline from local_dir instead of repo_id
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
    def generate(self, data: Annotated[dict, bentoml.validators.DataframeSchema()]) -> Annotated[Path, bentoml.validators.ContentType("image/png")]:
        import torch

        prompt = data["prompt"]
        trigger_word = data.get("trigger_word", "")

        # Switch LoRA adapter if needed
        if trigger_word in self.lora2_triggers and self.current_adapter != "flux-ghibsky":
            self.pipe.set_adapters(["flux-ghibsky"], adapter_weights=[0.8])
            self.current_adapter = "flux-ghibsky"
        elif trigger_word in self.lora1_triggers and self.current_adapter != "open-image-preferences":
            self.pipe.set_adapters(["open-image-preferences"], adapter_weights=[1.0])
            self.current_adapter = "open-image-preferences"

        # Compile components for performance
        self.pipe.text_encoder = torch.compile(self.pipe.text_encoder, fullgraph=False, mode="reduce-overhead")
        self.pipe.text_encoder_2 = torch.compile(self.pipe.text_encoder_2, fullgraph=False, mode="reduce-overhead")
        self.pipe.vae = torch.compile(self.pipe.vae, fullgraph=False, mode="reduce-overhead")

        with torch.no_grad():
            with vram_monitor(tag="Image Generation"):
                image = self.pipe(
                    prompt=prompt,
                    height=1024,
                    width=1024,
                    guidance_scale=3.5,
                    num_inference_steps=28,
                    max_sequence_length=512,
                ).images[0]
            return save_image(image)