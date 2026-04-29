from simple_speech2text import ASR
from simple_llm import llm
import gradio as gr
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.llms import HuggingFaceHub
import torch
import os

#######------------- Prompt Template-------------####

# This template is structured based on LLAMA2. If you are using other LLMs, feel free to remove the tags
temp = """
<s><<SYS>>
List the key points with details from the context: 
[INST] The context : {context} [/INST] 
<</SYS>>
"""
# here is the simplified version of the prompt template
# temp = """
# List the key points with details from the context: 
# The context : {context} 
# """

pt = PromptTemplate(
    input_variables=["context"],
    template= temp)

prompt_to_LLAMA2 = LLMChain(llm=llm, prompt=pt)

with gr.Blocks() as demo:
    gr.Markdown(" 🎤 REDA Transcription App")

    choice = gr.Radio(
        choices=["📁 Upload", "🎙️ Record"],
        value="📁 Upload",
        label="Choose Input Method"
    )

    upload_input = gr.Audio(sources=["upload"], type="filepath", visible=True)
    mic_input = gr.Audio(sources=["microphone"], type="filepath", visible=False)

    # Show/hide based on choice
    def toggle(choice):
        if choice == "📁 Upload":
            return gr.update(visible=True), gr.update(visible=False)
        else:
            return gr.update(visible=False), gr.update(visible=True)

    choice.change(fn=toggle, inputs=choice, outputs=[upload_input, mic_input])

    submit_btn = gr.Button("Submit")
    output = gr.Textbox(label="Result")

    def fgd_combined(upload, mic, choice):
        speech = upload if choice == "📁 Upload" else mic
        if speech is None:
            return "⚠️ Please provide an audio file first."
        transcrib = ASR(speech)
        result = prompt_to_LLAMA2.run(transcrib)
        return result

    submit_btn.click(
        fn=fgd_combined,
        inputs=[upload_input, mic_input, choice],
        outputs=output
    )

demo.launch(debug=True, share=True)