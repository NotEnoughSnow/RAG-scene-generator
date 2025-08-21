import torch
from diffusers import (
    StableDiffusionPipeline,
)

import logging
logging.basicConfig(level=logging.INFO)

import torch
import torch_directml

from diffusers import (
    EulerAncestralDiscreteScheduler,
    EulerDiscreteScheduler,
    DPMSolverMultistepScheduler,
    DPMSolverSinglestepScheduler,
    DPMSolverSDEScheduler,
    LMSDiscreteScheduler,
    HeunDiscreteScheduler,
    DDIMScheduler,
    UniPCMultistepScheduler
)


class Generator:

    def __init__(self):

        self.dml = torch_directml.device()

        #TODO change paths to dynamic

        checkpoint_path = "D:\\SD\\checkpoints\\dreamshaper_8.safetensors"
        # slora1_path = "D:\\SD\\lora\\swamp.safetensors"
        lora2_path = "D:\\SD\\lora\\tavern_scenes.safetensors"

        self.pipe = self.load_model(checkpoint_path)

        # dictionary of samplers
        samplers = {
            "euler_a": EulerAncestralDiscreteScheduler.from_config(self.pipe.scheduler.config),
            "euler": EulerDiscreteScheduler.from_config(self.pipe.scheduler.config),
            "dpmpp_2m": DPMSolverMultistepScheduler.from_config(self.pipe.scheduler.config),
            "dpmpp_sde": DPMSolverSDEScheduler.from_config(self.pipe.scheduler.config),
            "dpmpp_singlestep": DPMSolverSinglestepScheduler.from_config(self.pipe.scheduler.config),
            "lms": LMSDiscreteScheduler.from_config(self.pipe.scheduler.config),
            "heun": HeunDiscreteScheduler.from_config(self.pipe.scheduler.config),
            "ddim": DDIMScheduler.from_config(self.pipe.scheduler.config),
            "unipc": UniPCMultistepScheduler.from_config(self.pipe.scheduler.config),
        }

        # choose a sampler here
        chosen_sampler = "euler_a"  # for example
        self.pipe.scheduler = samplers[chosen_sampler]

        self.load_loras(
            lora_paths=[lora2_path],
            lora_scales=[1],
        )

        # Load an embedding (Textual Inversion .pt/.bin file you downloaded)
        self.pipe.load_textual_inversion("D:\\SD\\embedding\\easynegative.safetensors")


    def load_model(self, checkpoint_path):
        # Load SDXL checkpoint
        pipe = StableDiffusionPipeline.from_single_file(checkpoint_path, torch_dtype=torch.float16)
        pipe.to(self.dml)

        # Replace scheduler with Euler A
        pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(pipe.scheduler.config)

        return pipe

    def load_loras(
            self,
            lora_paths=None,
            lora_scales=None,
    ):
        # Load LoRAs if provided
        if lora_paths is None:
            lora_paths = []
        for i, lora_path in enumerate(lora_paths):
            self.pipe.load_lora_weights(lora_path)
            scale = lora_scales[i] if i < len(lora_scales) else 1.0
            self.pipe.fuse_lora(lora_scale=scale)


    def generate(self,
                 prompt="tavern",
                 negative_prompt="",
                 steps=55,
                 cfg_scale=7,
                 ):
        # generator = torch.Generator(device="cuda").manual_seed(seed)

        # Generate image
        image = self.pipe(
            prompt=prompt,
            negative_prompt=negative_prompt,
            num_inference_steps=steps,
            guidance_scale=cfg_scale,
            # generator=generator,
            # added_cond_kwargs=added_cond_kwargs,
        ).images[0]

        return image



