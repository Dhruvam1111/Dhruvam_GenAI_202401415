import torch
import gradio as gr
import json
from transformers import pipeline

model_path = "Models/models--facebook--nllb-200-distilled-600M/snapshots/f8d333a098d19b4fd9a8b18f94170487ad3f821d"
text_translator = pipeline("translation", model=model_path, torch_dtype=torch.bfloat16)

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

def translate_text(text, source_language, target_language):
    src_code, target_code = FLORES_code(source_language, target_language)
    if src_code == "Language not found in the dataset." or not src_code or not target_code:
        return "Language not found in the dataset."
    translation = text_translator(text, src_lang=src_code, tgt_lang=target_code)
    return translation[0]["translation_text"]

gr.close_all()

with gr.Blocks() as demo:
    gr.Markdown("# 🌐 Multilingual Translator")
    gr.Markdown("## Translate text between multiple languages.")

    with gr.Row():
        input_len_slider = gr.Slider(minimum=1, maximum=50, value=10, label="Input Textbox Height (Lines)")
        output_len_slider = gr.Slider(minimum=1, maximum=50, value=10, label="Output Textbox Height (Lines)")

    text_input = gr.Textbox(label="Input text to translate", lines=10, placeholder="Enter text...")
    src_dropdown = gr.Dropdown(label="Select source language", choices=[entry["Language"] for entry in data], value=None)
    tgt_dropdown = gr.Dropdown(label="Select target language", choices=[entry["Language"] for entry in data], value=None)
    output_box = gr.Textbox(label="Translation", lines=10, placeholder="Translation will appear here...")
    translate_btn = gr.Button("Translate")

    translate_btn.click(translate_text, inputs=[text_input, src_dropdown, tgt_dropdown], outputs=output_box)
    input_len_slider.change(lambda n: gr.update(lines=n), inputs=input_len_slider, outputs=text_input)
    output_len_slider.change(lambda n: gr.update(lines=n), inputs=output_len_slider, outputs=output_box)

demo.launch()
