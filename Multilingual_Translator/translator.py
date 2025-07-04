import torch
import gradio as gr
import json
import datetime
from transformers import pipeline

# Load model
text_translator = pipeline("translation", model="facebook/nllb-200-distilled-600M", torch_dtype=torch.bfloat16)

# Load language list
try:
    with open("Files/languages.json", "r", encoding="utf-8") as file:
        data = json.load(file)
except FileNotFoundError:
    print("File 'languages.json' not found.")
    data = []
except json.JSONDecodeError:
    print("Failed to decode JSON file.")
    data = []

# Get FLORES codes
def FLORES_code(source_language, target_language):
    src_code = tgt_code = None
    for entry in data:
        if entry["Language"].lower() == (source_language or "").lower():
            src_code = entry["FLORES-200 code"]
        if entry["Language"].lower() == (target_language or "").lower():
            tgt_code = entry["FLORES-200 code"]
    if src_code and tgt_code:
        return src_code, tgt_code
    return "Language not found in the dataset.", None

# Translate function
def translate_text(text, source_language, target_language):
    src_code, target_code = FLORES_code(source_language, target_language)
    if src_code == "Language not found in the dataset." or not src_code or not target_code:
        return "Language not found in the dataset."
    translation = text_translator(text, src_lang=src_code, tgt_lang=target_code)
    return translation[0]["translation_text"]

# Save translation to file
def save_translation(text):
    filename = f"translation_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
    with open(filename, "w", encoding="utf-8") as f:
        f.write(text)
    return filename

# Clear all fields
def clear_fields():
    return "", None, None, ""

gr.close_all()

# Interface
with gr.Blocks() as demo:
    gr.Markdown("# 🌐 Multilingual Translator")
    gr.Markdown("## Translate text between multiple languages.")

    # Line height sliders
    with gr.Row():
        input_len_slider = gr.Slider(minimum=1, maximum=50, value=5, label="Input Textbox Height (Lines)")
        output_len_slider = gr.Slider(minimum=1, maximum=50, value=5, label="Output Textbox Height (Lines)")

    # Main translation row
    with gr.Row():
        with gr.Column(scale=1):
            src_dropdown = gr.Dropdown(
                label="Select Source Language",
                choices=[entry["Language"] for entry in data],
                value=None
            )
            tgt_dropdown = gr.Dropdown(
                label="Select Destination Language",
                choices=[entry["Language"] for entry in data],
                value=None
            )
            translate_btn = gr.Button("🌍 Translate", scale=0)

        with gr.Column(scale=3):
            text_input = gr.Textbox(
                label="Input Text",
                lines=5,
                placeholder="Enter text to translate..."
            )
            output_box = gr.Textbox(
                label="Translated Text",
                lines=5,
                placeholder="Translation will appear here..."
            )

    # Action buttons
    with gr.Row():
        save_btn = gr.Button("💾 Save Translation")
        clear_btn = gr.Button("🧹 Clear All")
        file_output = gr.File(label="Download File")

    # Function bindings
    translate_btn.click(translate_text, inputs=[text_input, src_dropdown, tgt_dropdown], outputs=output_box)
    input_len_slider.change(lambda n: gr.update(lines=n), inputs=input_len_slider, outputs=text_input)
    output_len_slider.change(lambda n: gr.update(lines=n), inputs=output_len_slider, outputs=output_box)
    save_btn.click(save_translation, inputs=output_box, outputs=file_output)
    clear_btn.click(clear_fields, outputs=[text_input, src_dropdown, tgt_dropdown, output_box])

demo.launch()
