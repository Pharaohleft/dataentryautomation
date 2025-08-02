import gradio as gr
from transformers import pipeline, set_seed

generator = pipeline("text-generation", model="distilgpt2")
set_seed(42)

def chat(input_text):
    prompt = f"The following is an emotionally volatile reply to: {input_text}\nResponse:"
    result = generator(prompt, max_length=25, num_return_sequences=1)
    return result[0]['generated_text'].replace(prompt, '').strip()

gr.Interface(fn=chat, inputs="text", outputs="text", title="Ex Chatbot").launch()
