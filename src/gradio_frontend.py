import gradio as gr
import requests
from PIL import Image
from io import BytesIO

# Change this to wherever your BentoML service is running
BENTO_URL = "http://localhost:3000/generate"  # or your deployed endpoint

def generate_image(trigger_word, prompt):
    payload = {
        "trigger_word": trigger_word,
        "prompt": prompt
    }

    # Send request to BentoML API
    response = requests.post(BENTO_URL, json=payload)

    if response.status_code == 200:
        # Convert binary image to PIL
        img = Image.open(BytesIO(response.content))
        return img
    else:
        raise gr.Error(f"Error {response.status_code}: {response.text}")

# Build Gradio UI
with gr.Blocks() as demo:
    gr.Markdown(
    """
    <p style="font-size:20px">
    🎨 Flux Fast LoRA-Hotswap:
    This application uses **torch.compile**, **BitsAndBytes**, and **PEFT LoRA** to enable the 
    Flux.1 [dev] model for blazing fast text-to-image generation.  

    
    **Base model:** [black-forest-labs/FLUX.1-dev](https://huggingface.co/black-forest-labs/FLUX.1-dev)  
    **LoRA 1:** [data-is-better-together/open-image-preferences-v1-flux-dev-lora](https://huggingface.co/data-is-better-together/open-image-preferences-v1-flux-dev-lora)  
    **LoRA 2:** [aleksa-codes/flux-ghibsky-illustration](https://huggingface.co/aleksa-codes/flux-ghibsky-illustration)

    Trigger words for LoRA 1: ["Cinematic", "Photographic", "Anime", "Manga", "Digital art", "Pixel art", "Fantasy art", 
                                "Neonpunk", "3D Model", “Painting”, “Animation” “Illustration”]

    Trigger words for LoRA 2: ["GHIBSKY"]

    Inspired by the following [blog](https://huggingface.co/blog/lora-fast)
    </p>
    """
)

    with gr.Row():
        trigger_word = gr.Textbox(label="Trigger Word", placeholder="e.g. GHIBSKY,Cinematic")
        prompt = gr.Textbox(label="Prompt", placeholder="Describe your image...")

    generate_btn = gr.Button("Generate")
    output_image = gr.Image(label="Generated Image")

    generate_btn.click(
        fn=generate_image,
        inputs=[trigger_word, prompt],
        outputs=output_image
    )

if __name__ == "__main__":
    demo.launch(share=True, debug=True)
