import torch
from diffusers import StableDiffusionPipeline

import logging
logging.basicConfig(level=logging.INFO)

import torch
import torch_directml


class Generator:

    def __init__(self):


        dml = torch_directml.device()

        # --- SETTINGS ---
        base_model = "runwayml/stable-diffusion-v1-5"  # SD1.5 model
        self.lora_path = "./SD"  # Local folder or HF repo
        lora_file = "FantasyClassics.safetensors"  # LoRA file name

        # --- LOAD BASE MODEL ---
        self.pipe = StableDiffusionPipeline.from_pretrained(
            base_model,
            torch_dtype=torch.float16
        )
        self.pipe = self.pipe.to(dml)

    def load_lora(self, lora_file):

        self.pipe.load_lora_weights(self.lora_path, weight_name=lora_file)
        self.pipe.fuse_lora(lora_scale=0.8)  # Adjust 0.0-1.0 for LoRA strength

    def generate(self, prompt):
        image = self.pipe(prompt, guidance_scale=7.5, num_inference_steps=25).images[0]

        return image



