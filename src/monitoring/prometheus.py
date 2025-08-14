import os
import time
from contextlib import contextmanager

import psutil
import torch
from prometheus_client import Gauge, Histogram, start_http_server

# Prometheus metrics
GEN_TIME = Histogram("image_generation_seconds", "Time to generate image", ["adapter"])
VRAM_USED = Gauge("image_generation_vram_mb", "VRAM used after generation", ["adapter"])
VRAM_PEAK = Gauge("image_generation_vram_peak_mb", "Peak VRAM during generation", ["adapter"])
CPU_RAM_USED = Gauge("image_generation_cpu_ram_mb", "CPU RAM used after generation", ["adapter"])
PROMPT_LENGTH = Histogram(
    "image_generation_prompt_length", "Length of the prompt in characters", ["adapter"]
)


@contextmanager
def vram_monitor(tag="Run", adapter="unknown", prompt: str = ""):
    """
    Context manager to measure VRAM, CPU RAM, execution time, and prompt length,
    and expose them as Prometheus metrics.
    """
    if torch.cuda.is_available():
        torch.cuda.synchronize()

    start_time = time.time()

    # Capture initial CPU RAM
    process = psutil.Process()
    start_cpu_mem = process.memory_info().rss / (1024**2)

    # Capture prompt length
    prompt_len = len(prompt) if prompt else 0
    PROMPT_LENGTH.labels(adapter=adapter).observe(prompt_len)

    yield  # Execute wrapped code

    if torch.cuda.is_available():
        torch.cuda.synchronize()
        end_vram = torch.cuda.memory_allocated() / (1024**2)
        peak_vram = torch.cuda.max_memory_allocated() / (1024**2)
        torch.cuda.reset_peak_memory_stats()
    else:
        end_vram = 0
        peak_vram = 0

    end_cpu_mem = process.memory_info().rss / (1024**2)
    cpu_ram_used = end_cpu_mem - start_cpu_mem

    elapsed_time = time.time() - start_time

    # Emit Prometheus metrics
    GEN_TIME.labels(adapter=adapter).observe(elapsed_time)
    VRAM_USED.labels(adapter=adapter).set(end_vram)
    VRAM_PEAK.labels(adapter=adapter).set(peak_vram)
    CPU_RAM_USED.labels(adapter=adapter).set(cpu_ram_used)

    # Log for debugging
    print(
        f"[{tag}] Adapter={adapter} | PromptLen={prompt_len} chars "
        f"| Time={elapsed_time:.2f}s | VRAM={end_vram:.2f} MB | Peak={peak_vram:.2f} MB "
        f"| CPU RAM={cpu_ram_used:.2f} MB"
    )


# Start Prometheus server
PROM_PORT = int(os.environ.get("PROMETHEUS_PORT", 8000))
start_http_server(PROM_PORT)
print(f"Prometheus metrics available at http://localhost:{PROM_PORT}")
