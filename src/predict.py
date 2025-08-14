from __future__ import annotations

import os
import uuid
import time
import psutil
from pathlib import Path
from contextlib import contextmanager
from typing import Annotated

import bentoml
from PIL import Image  # Import early so type hints work

hf_token = os.environ.get("HF_TOKEN")

@contextmanager
def vram_monitor(tag="Run"):
    """Context manager to log VRAM & CPU RAM usage."""
    import torch
    torch.cuda.synchronize()
    start_time = time.time()

    start_vram = torch.cuda.memory_allocated() / (1024 ** 2)
    start_vram_reserved = torch.cuda.memory_reserved() / (1024 ** 2)
    start_cpu_mem = psutil.Process().memory_info().rss / (1024 ** 2)

    yield

    torch.cuda.synchronize()
    end_time = time.time()

    end_vram = torch.cuda.memory_allocated() / (1024 ** 2)
    end_vram_reserved = torch.cuda.memory_reserved() / (1024 ** 2)
    peak_vram = torch.cuda.max_memory_allocated() / (1024 ** 2)
    end_cpu_mem = psutil.Process().memory_info().rss / (1024 ** 2)

    print(f"\n[{tag}] Performance Report:")
    print(f"Elapsed time: {end_time - start_time:.2f} sec")
    print(f"VRAM Allocated: {start_vram:.2f} MB → {end_vram:.2f} MB")
    print(f"VRAM Reserved:  {start_vram_reserved:.2f} MB → {end_vram_reserved:.2f} MB")
    print(f"Peak VRAM:      {peak_vram:.2f} MB")
    print(f"CPU RAM:        {start_cpu_mem:.2f} MB → {end_cpu_mem:.2f} MB\n")

    torch.cuda.reset_peak_memory_stats()


def save_image(image: Image.Image, output_dir: Path = Path("/tmp")) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{uuid.uuid4().hex}.png"
    image.save(output_path)
    return output_path


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

        # Lazily load to avoid build-time GPU needs
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