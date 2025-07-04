import torch 
import gradio as gr 
# Use a pipeline as a high-level helper
from transformers import pipeline

# Load model for text summarization
text_summary = pipeline("summarization", model="sshleifer/distilbart-cnn-12-6", torch_dtype=torch.bfloat16)

# Function to summarize text
def summarize (text):
    output = text_summary(text)
    return output[0]['summary_text']

gr.close_all()

# Interface for text summarization
app = gr.Interface(fn=summarize,inputs=gr.Textbox(label="Plug in your text here to summarize", lines=10, placeholder="Enter text to summarize..."), outputs=gr.Textbox(label="Summary", lines=5, placeholder="Summary will appear here..."), title="Text-Summarizer", description="This application summarizes the input text.", theme="default")
app.launch()