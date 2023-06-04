from pathlib import Path
import gradio as gr
from modules import utils
from modules import shared
from modules.models import unload_model
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from pathlib import Path
import torch

params = {
    "display_name": "Merge",
    "is_tab": True,
}

refresh_symbol = '\U0001f504'  # 🔄

class ToolButton(gr.Button, gr.components.FormComponent):
    """Small button with single emoji as text, fits inside gradio forms"""

    def __init__(self, **kwargs):
        super().__init__(variant="tool", **kwargs)

    def get_block_name(self):
        return "button"


def create_refresh_button(refresh_component, refresh_method, refreshed_args, elem_id):
    def refresh():
        refresh_method()
        args = refreshed_args() if callable(refreshed_args) else refreshed_args

        for k, v in args.items():
            setattr(refresh_component, k, v)

        return gr.update(**(args or {}))

    refresh_button = ToolButton(value=refresh_symbol, elem_id=elem_id)
    refresh_button.click(
        fn=refresh,
        inputs=[],
        outputs=[refresh_component]
    )
    return refresh_button

def process_merge(model_name, peft_model_name, output_dir):
    
    base_model_name_or_path = Path(f'{shared.args.model_dir}/{model_name}')
    peft_model_path = Path(f'{shared.args.lora_dir}/{peft_model_name}')
    device_arg = { 'device_map': 'auto' }
    print(f"Unloading model")
    unload_model()

    print(f"Loading base model: {base_model_name_or_path}")
  
    try:
        base_model = AutoModelForCausalLM.from_pretrained(
            base_model_name_or_path,
            return_dict=True,
            torch_dtype=torch.float16,
            device_map={'': 0})    
    except Exception as e:
        print(f"\033[91mError in AutoModelForCausalLM: \033[0;37;0m\n {e}" )
        print(f"Merge failed")
        return
    
    print(f"Loading PEFT: {peft_model_path}")
    try:
        model = PeftModel.from_pretrained(base_model, peft_model_path, torch_dtype=torch.float16, device_map={'': 0})
    except Exception as e:
        print(f"\033[91mError initializing PeftModel:  \033[0;37;0m\n{e}")
        print(f"Merge failed")
        return

    print(f"Running merge_and_unload - WAIT untill you see the Model saved message")
    model = model.merge_and_unload()
    try:
        tokenizer = AutoTokenizer.from_pretrained(base_model_name_or_path)
    except Exception as e:
        print(f"\033[91mError in AutoTokenizer:  \033[0;37;0m\n{e}")
        print(f"Merge failed")
        return
    model.save_pretrained(f"{output_dir}")
    tokenizer.save_pretrained(f"{output_dir}")
    print(f"Model saved to {output_dir}")
    print(f"**** DONE ****")


def ui():
    model_name = "None"
    lora_names = "None"
    with gr.Accordion("Merge Model with Lora", open=True):
        
        with gr.Row():
               
            with gr.Column():
                with gr.Row():
                    gr_modelmenu = gr.Dropdown(choices=utils.get_available_models(), value=model_name, label='Model 16-bit HF only')
                    create_refresh_button(gr_modelmenu, lambda: None, lambda: {'choices': utils.get_available_models()}, 'refresh-button')

            with gr.Column():
                with gr.Row():
                    gr_loramenu = gr.Dropdown(multiselect=False, choices=utils.get_available_loras(), value=lora_names, label='LoRA')
                    create_refresh_button(gr_loramenu, lambda: None, lambda: {'choices': utils.get_available_loras(), 'value': lora_names}, 'refresh-button')

        output_dir = gr.Textbox(label='Output Dir', info='The folder name of your merge (relative to text-generation-webui)')
        gr_apply = gr.Button(value='Do Merge')    
        gr_apply.click(process_merge, inputs=[gr_modelmenu, gr_loramenu,output_dir])        
