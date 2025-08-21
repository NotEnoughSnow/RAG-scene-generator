from diffusers import (
    StableDiffusionPipeline,
    StableDiffusionXLPipeline,
    FluxPipeline
)
import torch
import torch_directml
from pathlib import Path


class Generator:
    def __init__(self, base_model="runwayml/stable-diffusion-v1-5", model_type="sd15", local_file=False):
        self.dml = torch_directml.device()
        self.base_model = base_model
        self.model_type = model_type
        self.local_file = local_file
        self.lora_path = "./backend/resources/"

        self.pipe = self.load_pipeline()

    def load_pipeline(self):
        if self.local_file:
            model_path = Path(self.base_model)
            if not model_path.exists():
                raise FileNotFoundError(f"Model file not found: {model_path}")
            print("from file")
            if self.model_type == "sd15":
                pipe = StableDiffusionPipeline.from_single_file(model_path, torch_dtype=torch.float16)
            elif self.model_type == "sdxl":
                pipe = StableDiffusionXLPipeline.from_single_file(model_path, torch_dtype=torch.float16)
            else:
                pipe = FluxPipeline.from_single_file(model_path, torch_dtype=torch.float16)

        else:
            print("from pretrained")
            if self.model_type == "sd15":
                pipe = StableDiffusionPipeline.from_pretrained(self.base_model, torch_dtype=torch.float16)
            elif self.model_type == "sdxl":
                pipe = StableDiffusionXLPipeline.from_pretrained(self.base_model, torch_dtype=torch.float16)
            else:
                pipe = FluxPipeline.from_pretrained(self.base_model, torch_dtype=torch.bfloat16)

        return pipe.to(self.dml)

    def load_lora(self, lora_file, scale=0.8):
        self.pipe.load_lora_weights(self.lora_path, weight_name=lora_file)
        self.pipe.fuse_lora(lora_scale=scale)

    def generate(self, prompt, guidance_scale=7.5, steps=25, width=None, height=None):
        kwargs = dict(
            prompt=prompt,
            guidance_scale=guidance_scale,
            num_inference_steps=steps
        )

        if width and height:
            kwargs["width"] = width
            kwargs["height"] = height

        image = self.pipe(**kwargs).images[0]
        return image

prompt = "big castle"

#gen = Generator(base_model="black-forest-labs/FLUX.1-dev", model_type="auto")

gen = Generator(base_model="runwayml/stable-diffusion-v1-5", model_type="sd15")

image = gen.generate(prompt=prompt)

image.save("output.png")