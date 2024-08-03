
import gradio as gr
from src.models.llama_model import OptimizedLlamaModel
from src.nlp.command_generator import generate_command, validate_command
from src.system_integration.sandbox import safe_system_call

model = OptimizedLlamaModel()

def process_input(user_input):
    ai_response = model(user_input)
    suggested_command = generate_command(user_input)
    return ai_response, suggested_command

def execute_command(command):
    if validate_command(command):
        return safe_system_call(command)
    else:
        return "Error: Unsafe command"

def create_collaborative_interface():
    with gr.Blocks() as interface:
        chatbot = gr.Chatbot()
        msg = gr.Textbox()
        clear = gr.Button("Clear")

        def user(user_message, history):
            ai_response, suggested_command = process_input(user_message)
            return "", history + [[user_message, f"AI: {ai_response}\n\nSuggested Command: {suggested_command}"]]

        def bot(history):
            bot_message = history[-1][1]
            history[-1][1] = bot_message
            return history

        msg.submit(user, [msg, chatbot], [msg, chatbot], queue=False).then(bot, chatbot, chatbot)
        clear.click(lambda: None, None, chatbot, queue=False)

    return interface

def launch_collaborative_interface():
    interface = create_collaborative_interface()
    interface.launch(share=False, inbrowser=True)
