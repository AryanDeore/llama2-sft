"""Gradio web UI for story generation using fine-tuned Llama 2."""

import os

import gradio as gr
import torch

from generate import generate_story
from models.llama2 import Transformer

# HF Hub repo for the SFT model
HF_REPO_ID = os.environ.get("HF_REPO_ID", "0rn0/llama2-15m-tinystories-sft")

# Auto-detect device
if torch.cuda.is_available():
    DEVICE = "cuda"
elif torch.backends.mps.is_available():
    DEVICE = "cpu"
else:
    DEVICE = "cpu"

# Load model from HF Hub using PyTorchModelHubMixin
print(f"Loading model from {HF_REPO_ID}...")
model = Transformer.from_pretrained(HF_REPO_ID)
model = model.to(DEVICE)
model.eval()
print(f"Model loaded on {DEVICE}! Ready to generate stories.\n")


def generate(topic, ending, temperature):
    """Generate a story based on user inputs."""
    if not topic.strip():
        return "Please enter a topic for the story."

    story = generate_story(
        model=model,
        topic=topic.strip(),
        ending=ending.lower(),
        max_new_tokens=192,
        temperature=temperature,
        top_k=50,
        device=DEVICE,
    )

    return story


CUSTOM_CSS = """
@import url('https://fonts.googleapis.com/css2?family=Baloo+2:wght@600;800&family=Nunito:wght@400;600&display=swap');

.app-title {
    font-family: 'Baloo 2', cursive !important;
    font-size: 2.8rem !important;
    font-weight: 800 !important;
    color: #d15f00 !important;
    text-align: center;
    margin: 0 !important;
    line-height: 1.2;
}

.app-subtitle {
    font-family: 'Nunito', sans-serif !important;
    font-size: 1.05rem !important;
    color: #666 !important;
    text-align: center;
    margin-top: 4px !important;
}

/* Card panels */
.card-panel {
    background: #ffffff !important;
    border-radius: 16px !important;
    box-shadow: 0 4px 16px rgba(0, 0, 0, 0.08) !important;
    padding: 28px !important;
    border: 1.5px solid #f0e6d3 !important;
}

/* Nuke ALL grey backgrounds inside cards */
.card-panel .block,
.card-panel .form,
.card-panel .gap,
.card-panel > div,
.card-panel label,
.card-panel .wrap,
.card-panel .wrap.default,
.card-panel fieldset,
.card-panel .container {
    border: none !important;
    box-shadow: none !important;
    background: transparent !important;
    background-color: transparent !important;
    padding: 0 !important;
}

/* Spacing between components in the left card */
.card-panel .block + .block {
    margin-top: 20px !important;
}

/* Textbox */
.card-panel .gradio-textbox textarea {
    background-color: #ffffff !important;
    border: 1.5px solid #d0d0d0 !important;
    border-radius: 10px !important;
    padding: 10px 12px !important;
    font-family: 'Nunito', sans-serif !important;
    font-size: 0.95rem !important;
    margin-top: 6px !important;
}

.card-panel .gradio-textbox textarea:focus {
    border-color: #ff8c00 !important;
    box-shadow: 0 0 0 2px rgba(255, 140, 0, 0.15) !important;
    outline: none !important;
}

/* Component labels */
.card-panel .block > label > span,
.card-panel .block > span {
    font-family: 'Nunito', sans-serif !important;
    font-size: 0.95rem !important;
    font-weight: 600 !important;
    color: #444 !important;
    margin-bottom: 6px !important;
    display: block !important;
}

/* Slider number input */
.card-panel input[type='number'] {
    background-color: #ffffff !important;
    border: 1.5px solid #d0d0d0 !important;
    border-radius: 8px !important;
    padding: 4px 8px !important;
    font-family: 'Nunito', sans-serif !important;
}

/* Storybook output box */
.story-output textarea {
    font-family: 'Nunito', sans-serif !important;
    font-size: 1rem !important;
    line-height: 1.8 !important;
    background-color: #fffbf2 !important;
    border: 1.5px solid #e8d5b0 !important;
    border-radius: 12px !important;
    padding: 16px !important;
    color: #3b2f1e !important;
    margin-top: 6px !important;
    min-height: 300px !important;
}

.story-output .block > label > span {
    font-family: 'Baloo 2', cursive !important;
    font-size: 1rem !important;
    color: #d15f00 !important;
    font-weight: 700 !important;
}

/* Generate button */
.gen-btn {
    font-family: 'Baloo 2', cursive !important;
    font-size: 1.15rem !important;
    font-weight: 700 !important;
    border-radius: 12px !important;
    margin-top: 24px !important;
    padding: 10px 0 !important;
    background: linear-gradient(135deg, #ff8c00, #e05c00) !important;
    border: none !important;
    box-shadow: 0 4px 10px rgba(224, 92, 0, 0.35) !important;
    transition: transform 0.1s ease, box-shadow 0.1s ease !important;
}

.gen-btn:hover {
    transform: translateY(-1px) !important;
    box-shadow: 0 6px 14px rgba(224, 92, 0, 0.45) !important;
}

.gen-btn:active {
    transform: translateY(0px) !important;
}

/* Generate button - reduce font weight */
.gen-btn {
    font-family: 'Baloo 2', cursive !important;
    font-size: 1.15rem !important;
    font-weight: 600 !important;          /* was 700 */
    border-radius: 12px !important;
    margin-top: 24px !important;
    padding: 10px 0 !important;
    background: linear-gradient(135deg, #ff8c00, #e05c00) !important;
    border: none !important;
    box-shadow: 0 4px 10px rgba(224, 92, 0, 0.35) !important;
    transition: transform 0.1s ease, box-shadow 0.1s ease !important;
}

/* Reduce top padding inside both cards */
.card-panel {
    background: #ffffff !important;
    border-radius: 16px !important;
    box-shadow: 0 4px 16px rgba(0, 0, 0, 0.08) !important;
    padding: 16px 28px 28px 28px !important;   /* was 28px all around */
    border: 1.5px solid #f0e6d3 !important;
}


/* Footer */
.footer {
    position: fixed !important;
    bottom: 0 !important;
    left: 0 !important;
    right: 0 !important;
    text-align: center;
    padding: 16px;
    display: flex;
    justify-content: center;
    flex-wrap: wrap;
    gap: 20px;
    border-top: 1px solid #f0e6d3;
    background: white;
    z-index: 100;
    width: 100% !important;
    box-sizing: border-box !important;
}

.footer a {
    color: #bbb;
    transition: color 0.2s ease;
    display: flex;
    align-items: center;
    gap: 6px;
    font-family: 'Nunito', sans-serif;
    font-size: 0.85rem;
    text-decoration: none;
}

.footer svg {
    width: 18px;
    height: 18px;
    fill: #888;
    transition: fill 0.2s ease;
}

/* LinkedIn link */
.footer a:nth-child(1):hover {
    color: #0A66C2;
}

.footer a:nth-child(1):hover svg {
    fill: #0A66C2;
}

/* Website link */
.footer a:nth-child(2):hover {
    color: #D4864A;
}

.footer a:nth-child(2):hover svg {
    fill: #D4864A;
}

/* Gmail link */
.footer a:nth-child(3):hover {
    color: #EA4335;
}

.footer a:nth-child(3):hover svg {
    fill: #EA4335;
}

/* Add padding to gradio container so content doesn't hide behind fixed footer */
.gradio-container {
    padding-bottom: 100px !important;
}
"""

with gr.Blocks(title="Tiny Tales Llama") as demo:
    gr.HTML("<link rel='preconnect' href='https://fonts.googleapis.com'><link rel='preconnect' href='https://fonts.gstatic.com' crossorigin>")

    gr.Markdown("<br>")
    gr.HTML("<h1 class='app-title'>Tiny Tales Llama</h1>")
    gr.HTML("<p class='app-subtitle'>Generate short stories using a pre-trained and instruction fine-tuned Llama 2 15M model.</p>")
    gr.HTML(
        "<div style='text-align: center; display: flex; justify-content: center; flex-wrap: wrap; gap: 8px; margin: 12px 0 20px;'>"
        "<a href='https://github.com/AryanDeore/Tiny-Tales-GPT'><img src='https://img.shields.io/badge/GitHub-Pre%20Training%20(GPT2)-181717?logo=github&style=flat-square' style='height: clamp(20px, 3vw, 23px); border-radius: 6px;' /></a>"
        "<a href='https://github.com/AryanDeore/llama2-sft'><img src='https://img.shields.io/badge/GitHub-Instruction%20Fine%20Tuning-181717?logo=github&style=flat-square' style='height: clamp(20px, 3vw, 23px); border-radius: 6px;' /></a>"
        "<a href='https://huggingface.co/0rn0'><img src='https://img.shields.io/badge/HuggingFace-Collection-FFD21E?logo=huggingface&style=flat-square' style='height: clamp(20px, 3vw, 23px); border-radius: 6px;' /></a>"
        "</div>"
    )
    gr.Markdown("<br>")

    with gr.Row():
        with gr.Column(scale=1, elem_classes=["card-panel"]):
            topic = gr.Textbox(
                label="Generate a short story about:",
                placeholder="a brave knight on an adventure",
                lines=3,
            )
            ending = gr.Radio(
                choices=["Happy", "Sad"],
                label="With ending:",
                value="Happy",
            )
            temperature = gr.Slider(
                minimum=0.1,
                maximum=1.4,
                value=0.7,
                step=0.1,
                label="Temperature (creativity)",
                info="Low = predictable, High = creative",
            )
            submit_btn = gr.Button(
                "Generate Story",
                variant="primary",
                elem_classes=["gen-btn"],
            )

        with gr.Column(scale=1, elem_classes=["card-panel"]):
            output = gr.Textbox(
                label="Generated Story",
                lines=10,
                placeholder="Your story will appear here...",
                elem_classes=["story-output"],
            )

    # ↓ footer goes here, outside the Row
    gr.HTML("<div style='flex: 1;'></div>")
    gr.HTML(
        "<div class='footer'>"
        # LinkedIn
        "<a href='https://linkedin.com/in/aryandeore' target='_blank'>"
        "<svg viewBox='0 0 24 24' xmlns='http://www.w3.org/2000/svg'><path d='M20.447 20.452h-3.554v-5.569c0-1.328-.027-3.037-1.852-3.037-1.853 0-2.136 1.445-2.136 2.939v5.667H9.351V9h3.414v1.561h.046c.477-.9 1.637-1.85 3.37-1.85 3.601 0 4.267 2.37 4.267 5.455v6.286zM5.337 7.433a2.062 2.062 0 0 1-2.063-2.065 2.064 2.064 0 1 1 2.063 2.065zm1.782 13.019H3.555V9h3.564v11.452zM22.225 0H1.771C.792 0 0 .774 0 1.729v20.542C0 23.227.792 24 1.771 24h20.451C23.2 24 24 23.227 24 22.271V1.729C24 .774 23.2 0 22.222 0h.003z'/></svg>"
        "aryandeore"
        "</a>"
        # Website
        "<a href='https://www.aryandeore.ai' target='_blank'>"
        "<svg viewBox='0 0 24 24' xmlns='http://www.w3.org/2000/svg'><path d='M12 0C5.373 0 0 5.373 0 12s5.373 12 12 12 12-5.373 12-12S18.627 0 12 0zm-1 4.062V8H7.062A8.006 8.006 0 0 1 11 4.062zM4.062 13H8v3H5.674A7.953 7.953 0 0 1 4.062 13zm1.612 5H8v2.938A8.006 8.006 0 0 1 5.674 18zM11 19.938V16h3.326A8.006 8.006 0 0 1 11 19.938zM11 14v-3h2v3h-2zm0-5V4.062A8.006 8.006 0 0 1 14.938 8H11zm5 9v-3h2.326a7.953 7.953 0 0 1-2.326 3zm2.326-5H16v-3h3.938a7.953 7.953 0 0 1-1.612 3zM16 8V5.062A8.006 8.006 0 0 1 19.938 9H16zm-3-3.938V8h-2V4.062A8.006 8.006 0 0 1 13 4.062z'/></svg>"
        "aryandeore.ai"
        "</a>"
        # Email
        "<a href='mailto:aryandeore.work@gmail.com'>"
        "<svg viewBox='0 0 24 24' xmlns='http://www.w3.org/2000/svg'><path d='M24 5.457v13.909c0 .904-.732 1.636-1.636 1.636h-3.819V11.73L12 16.64l-6.545-4.91v9.273H1.636A1.636 1.636 0 0 1 0 19.366V5.457c0-.9.732-1.636 1.636-1.636h.749L12 10.638 21.615 3.82h.749c.904 0 1.636.737 1.636 1.636z'/></svg>"
        "Get in touch"
        "</a>"
        "</div>"
    )

    submit_btn.click(fn=generate, inputs=[topic, ending, temperature], outputs=output)
    topic.submit(fn=generate, inputs=[topic, ending, temperature], outputs=output)

if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",
        server_port=int(os.environ.get("PORT", 7860)),
        css=CUSTOM_CSS,
    )