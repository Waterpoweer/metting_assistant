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

def fgd(speech) :

    transcrib = ASR(speech)



    result = prompt_to_LLAMA2.run(transcrib)

    return result




audio_input = gr.Audio(sources="upload", type="filepath")  # Audio input
output_text = gr.Textbox()  # Text output

# Create the Gradio interface with the function, inputs, and outputs
iface = gr.Interface(fn=fgd, 
                     inputs=audio_input, outputs=output_text, 
                     title="Audio Transcription App",
                     description="Upload the audio file")

# Launch the Gradio app
iface.launch(debug = True  ,  share=True)