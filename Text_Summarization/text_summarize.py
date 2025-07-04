import torch # type: ignore
import gradio as gr # type: ignore
# Use a pipeline as a high-level helper
from transformers import pipeline

# text_summary = pipeline("summarization", model="sshleifer/distilbart-cnn-12-6", torch_dtype=torch.bfloat16)

model_path = ("Models/models--sshleifer--distilbart-cnn-12-6/snapshots/a4f8f3ea906ed274767e9906dbaede7531d660ff")

text_summary = pipeline("summarization", model=model_path, torch_dtype=torch.bfloat16)

# text = '''The cat (Felis catus), also referred to as the domestic cat or house cat, is a small domesticated carnivorous mammal. It is the only domesticated species of the family Felidae. Advances in archaeology and genetics have shown that the domestication of the cat occurred in the Near East around 7500 BC. It is commonly kept as a pet and working cat, but also ranges freely as a feral cat avoiding human contact. It is valued by humans for companionship and its ability to kill vermin. Its retractable claws are adapted to killing small prey species such as mice and rats. It has a strong, flexible body, quick reflexes, and sharp teeth, and its night vision and sense of smell are well developed. It is a social species, but a solitary hunter and a crepuscular predator.'''
# print(text_summary(text))

def summarize (text):
    output = text_summary(text)
    return output[0]['summary_text']

gr.close_all()
app = gr.Interface(fn=summarize,inputs=gr.Textbox(label="Plug in your text here to summarize", lines=10, placeholder="Enter text to summarize..."), outputs=gr.Textbox(label="Summary", lines=5, placeholder="Summary will appear here..."), title="Text-Summarizer", description="This application summarizes the input text.", theme="default")
app.launch()