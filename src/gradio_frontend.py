from io import BytesIO
from typing import Tuple

import gradio as gr
import requests  # type: ignore[import]
from PIL import Image

# Change this to wherever your BentoML service is running
BENTO_URL: str = "https://diffusers-project1.cloud.bentoml.com/"  # or your deployed endpoint


def generate_image(trigger_word: str, prompt: str) -> Image.Image:
    """Send prompt to BentoML API and return generated image."""
    payload: dict[str, str] = {"trigger_word": trigger_word, "prompt": prompt}
    response: requests.Response = requests.post(BENTO_URL, json=payload)

    if response.status_code == 200:
        img: Image.Image = Image.open(BytesIO(response.content))
        return img
    else:
        raise gr.Error(f"Error {response.status_code}: {response.text}")


# Build Gradio UI
with gr.Blocks() as demo:
    gr.Markdown(
        """
    <style>
        .lora-description {
            font-size: 18px;
            line-height: 1.6;
            color: #222;
            background-color: #fafafa;
            border-radius: 12px;
            padding: 20px 25px;
            box-shadow: 0 2px 8px rgba(0,0,0,0.05);
            max-width: 800px;
            margin: auto;
        }
        .lora-description h2 {
            font-size: 26px;
            margin-bottom: 10px;
            color: #444;
        }
        .lora-description a {
            color: #007acc;
            text-decoration: none;
        }
        .lora-description a:hover {
            text-decoration: underline;
        }
        .lora-description strong {
            color: #333;
        }
    </style>

    <div class="lora-description">
        <h2>🎨 Flux Fast LoRA-Hotswap</h2>
        <p>
            This application combines <strong>torch.compile</strong>, <strong>BitsAndBytes</strong>, 
            and <strong>PEFT LoRA</strong> to optimize and serve the 
            <a href="https://huggingface.co/black-forest-labs/FLUX.1-dev" target="_blank">Flux.1 [dev]</a> model 
            for blazing fast, high-quality text-to-image generation.  
            By enabling <em>LoRA hotswapping</em>, you can instantly switch between multiple style adapters without 
            reloading the model.
        </p>
        <p>
            <strong>Base model:</strong> 
            <a href="https://huggingface.co/black-forest-labs/FLUX.1-dev" target="_blank">
            black-forest-labs/FLUX.1-dev</a><br>
            <strong>LoRA 1:</strong> 
            <a href="https://huggingface.co/data-is-better-together/open-image-preferences-v1-flux-dev-lora" target="_blank">
            data-is-better-together/open-image-preferences-v1-flux-dev-lora</a><br>
            <strong>LoRA 2:</strong> 
            <a href="https://huggingface.co/aleksa-codes/flux-ghibsky-illustration" target="_blank">
            aleksa-codes/flux-ghibsky-illustration</a>
        </p>
        <p>
            <strong>Trigger words for LoRA 1:</strong> Cinematic, Photographic, Anime, Manga, Digital art, Pixel art, Fantasy art, 
            Neonpunk, 3D Model, Painting, Animation, Illustration<br>
            <strong>Trigger words for LoRA 2:</strong> GHIBSKY
        </p>
        <p>
            Inspired by the following 
            <a href="https://huggingface.co/blog/lora-fast" target="_blank">blog post</a>.
        </p>
    </div>
    """
    )

    with gr.Row():
        trigger_word = gr.Textbox(label="Trigger Word", placeholder="e.g. GHIBSKY,Cinematic")
        prompt = gr.Textbox(label="Prompt", placeholder="Describe your image...")

    generate_btn = gr.Button("Generate")
    download_btn = gr.Button("Download")  # Separate download button

    output_image = gr.Image(label="Generated Image")
    download_file = gr.File(label="Download Image", visible=False)  # Hidden until download
    image_path_state = gr.State()  # Store image internally

    # Generate image and store path
    def generate_and_store(trigger_word: str, prompt: str) -> Tuple[Image.Image, Image.Image]:
        image: Image.Image = generate_image(trigger_word, prompt)
        return image, image

    generate_btn.click(
        fn=generate_and_store,
        inputs=[trigger_word, prompt],
        outputs=[output_image, image_path_state],
    )

    # Trigger download separately
    def provide_download(image: Image.Image) -> Image.Image:
        return image

    download_btn.click(fn=provide_download, inputs=image_path_state, outputs=download_file)

if __name__ == "__main__":
    demo.launch(server_port=7860, share=True, debug=True)
