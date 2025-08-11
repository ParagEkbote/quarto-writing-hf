import time
import torch
import psutil
from contextlib import contextmanager
from prometheus_client import Histogram, Gauge

# Prometheus metrics
GEN_TIME = Histogram(
    "image_generation_seconds",
    "Time to generate image",
    ["adapter"]
)
VRAM_USED = Gauge(
    "image_generation_vram_mb",
    "VRAM used after generation",
    ["adapter"]
)
VRAM_PEAK = Gauge(
    "image_generation_vram_peak_mb",
    "Peak VRAM during generation",
    ["adapter"]
)
PROMPT_LENGTH = Histogram(
    "image_generation_prompt_length",
    "Length of the prompt in characters",
    ["adapter"]
)

@contextmanager
def vram_monitor(tag="Run", adapter="unknown", prompt: str = ""):
    """
    Context manager to measure VRAM usage, execution time, and prompt length,
    and expose them as Prometheus metrics.
    """
    torch.cuda.synchronize()
    start_time = time.time()

    # Capture prompt length
    prompt_len = len(prompt) if prompt else 0
    PROMPT_LENGTH.labels(adapter=adapter).observe(prompt_len)

    yield  # Execute the code

    torch.cuda.synchronize()
    elapsed_time = time.time() - start_time
    end_vram = torch.cuda.memory_allocated() / (1024 ** 2)
    peak_vram = torch.cuda.max_memory_allocated() / (1024 ** 2)

    # Emit metrics
    GEN_TIME.labels(adapter=adapter).observe(elapsed_time)
    VRAM_USED.labels(adapter=adapter).set(end_vram)
    VRAM_PEAK.labels(adapter=adapter).set(peak_vram)

    # Reset peak stats for next run
    torch.cuda.reset_peak_memory_stats()

    # Also print for logs
    print(
        f"[{tag}] Adapter={adapter} | PromptLen={prompt_len} chars "
        f"| Time={elapsed_time:.2f}s | VRAM={end_vram:.2f} MB | Peak={peak_vram:.2f} MB"
    )
