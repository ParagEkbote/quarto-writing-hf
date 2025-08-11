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
    gr.Markdown("## 🎨 Flux Fast LoRA:")

    with gr.Row():
        trigger_word = gr.Textbox(label="Trigger Word", placeholder="e.g. GHIBSKY")
        prompt = gr.Textbox(label="Prompt", placeholder="Describe your image...")

    generate_btn = gr.Button("Generate")
    output_image = gr.Image(label="Generated Image")

    generate_btn.click(
        fn=generate_image,
        inputs=[trigger_word, prompt],
        outputs=output_image
    )

if __name__ == "__main__":
    demo.launch()
