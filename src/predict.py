import os
import uuid
from pathlib import Path

import torch
import bentoml
from bentoml.io import JSON, File
from diffusers import DiffusionPipeline
from diffusers.quantizers import PipelineQuantizationConfig

from huggingface_hub import login
from PIL import Image

@bentoml.service(
    name="diffusers-fast-lora",
    traffic={"timeout": 300},
    envs=[{"name": "HF_TOKEN"}],
    resources={
        "gpu": 1,
        "gpu_type": "nvidia-a100-80gb"
    },
)

def save_image(image: Image.Image, output_dir: Path = Path("/tmp")) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{uuid.uuid4().hex}.png"
    image.save(output_path)
    return output_path


@bentoml.service(name="flux_lora_service", traffic={"timeout": 600})
class FluxLoRAService:
    @bentoml.on_startup
    def __init__(self):
        base_model_path = bentoml.models.get("flux_base").path
        lora_open_path = bentoml.models.get("lora_open_image").path
        lora_ghibsky_path = bentoml.models.get("lora_flux_ghibsky").path
        self.pipe = DiffusionPipeline.from_pretrained(
            "black-forest-labs/FLUX.1-dev",
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
        self.pipe.load_lora_weights(lora_open_path,
            "data-is-better-together/open-image-preferences-v1-flux-dev-lora",
            weight_name="pytorch_lora_weights.safetensors",
            adapter_name="open-image-preferences",
        )
        self.pipe.load_lora_weights(lora_ghibsky_path,
            "aleksa-codes/flux-ghibsky-illustration",
            weight_name="lora_v2.safetensors",
            adapter_name="flux-ghibsky",
        )
        self.current_adapter = "open-image-preferences"
        self.lora1_triggers = [
            "Cinematic", "Photographic", "Anime", "Manga", "Digital art",
            "Pixel art", "Fantasy art", "Neonpunk", "3D Model",
            "Painting", "Animation", "Illustration",
        ]
        self.lora2_triggers = ["GHIBSKY"]

    @bentoml.api(input=JSON(), output=File())
    def generate(self, data: dict) -> Path:
        prompt = data["prompt"]
        trigger_word = data["trigger_word"]

        if trigger_word in self.lora2_triggers and self.current_adapter != "flux-ghibsky":
            self.pipe.set_adapters(["flux-ghibsky"], adapter_weights=[0.8])
            self.current_adapter = "flux-ghibsky"
        elif trigger_word in self.lora1_triggers and self.current_adapter != "open-image-preferences":
            self.pipe.set_adapters(["open-image-preferences"], adapter_weights=[1.0])
            self.current_adapter = "open-image-preferences"

        self.pipe.text_encoder = torch.compile(self.pipe.text_encoder, fullgraph=False, mode="reduce-overhead")
        self.pipe.text_encoder_2 = torch.compile(self.pipe.text_encoder_2, fullgraph=False, mode="reduce-overhead")
        self.pipe.vae = torch.compile(self.pipe.vae, fullgraph=False, mode="reduce-overhead")

        with torch.no_grad():
            image = self.pipe(
                prompt=prompt,
                height=1024,
                width=1024,
                guidance_scale=3.5,
                num_inference_steps=28,
                max_sequence_length=512,
            ).images[0]
            return save_image(image)
