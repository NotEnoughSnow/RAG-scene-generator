import os
from urllib.parse import urlparse
import torch
from diffusers import EulerAncestralDiscreteScheduler
import requests
from diffusers import StableDiffusionPipeline

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

import torch_directml

DML = torch_directml.device()


def load_model(checkpoint_path):
    # Load SDXL checkpoint
    pipe = StableDiffusionPipeline.from_single_file(checkpoint_path, torch_dtype=torch.float16)
    pipe.to(DML)

    # Replace scheduler with Euler A
    pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(pipe.scheduler.config)

    return pipe

def load_loras(
    pipe,
    lora_paths=[],
    lora_scales=[],
):

    # Load LoRAs if provided
    for i, lora_path in enumerate(lora_paths):
        pipe.load_lora_weights(lora_path)
        scale = lora_scales[i] if i < len(lora_scales) else 1.0
        pipe.fuse_lora(lora_scale=scale)

    return pipe

def generate_image(pipe,
    prompt="tavern",
    negative_prompt="",
    steps=30,
    cfg_scale=3.5,
    seed=42,
):

    # Set seed
    #generator = torch.Generator(device="cuda").manual_seed(seed)

    # Generate image
    image = pipe(
        prompt=prompt,
        negative_prompt=negative_prompt,
        num_inference_steps=steps,
        guidance_scale=cfg_scale,
        #generator=generator,
        #added_cond_kwargs=added_cond_kwargs,
    ).images[0]

    # Save result
    output_path = "result.png"
    image.save(output_path)
    print(f"Image saved to {output_path}")
    return output_path


#checkpoint_path = download_checkpoint("https://civitai.com/api/download/models/654159?type=Model&format=SafeTensor&size=pruned&fp=fp16", "checkpoint.safetensors")
#lora1_path = download_lora("https://civitai.com/api/download/models/283697?type=Model&format=SafeTensor", "midjourney_lora.safetensors")
#lora2_path = download_lora("https://civitai.com/api/download/models/324583?type=Model&format=SafeTensor", "pixel.safetensors")

checkpoint_path = "D:\\SD\\checkpoints\\dreamshaper_8.safetensors"
#slora1_path = "D:\\SD\\lora\\swamp.safetensors"
lora2_path = "D:\\SD\\lora\\tavern_scenes.safetensors"


pipe = load_model(checkpoint_path)

# dictionary of samplers
samplers = {
    "euler_a": EulerAncestralDiscreteScheduler.from_config(pipe.scheduler.config),
    "euler": EulerDiscreteScheduler.from_config(pipe.scheduler.config),
    "dpmpp_2m": DPMSolverMultistepScheduler.from_config(pipe.scheduler.config),
    "dpmpp_sde": DPMSolverSDEScheduler.from_config(pipe.scheduler.config),
    "dpmpp_singlestep": DPMSolverSinglestepScheduler.from_config(pipe.scheduler.config),
    "lms": LMSDiscreteScheduler.from_config(pipe.scheduler.config),
    "heun": HeunDiscreteScheduler.from_config(pipe.scheduler.config),
    "ddim": DDIMScheduler.from_config(pipe.scheduler.config),
    "unipc": UniPCMultistepScheduler.from_config(pipe.scheduler.config),
}


# choose a sampler here
chosen_sampler = "euler_a"  # for example
pipe.scheduler = samplers[chosen_sampler]


pipe = load_loras(
    pipe=pipe,
    lora_paths=[lora2_path],
    lora_scales=[1],
    )



# Load an embedding (Textual Inversion .pt/.bin file you downloaded)
pipe.load_textual_inversion("D:\\SD\\embedding\\easynegative.safetensors")

"The paladin, Ser Ryam, stands like an unyielding statue beneath the corridor’s arch, his tarnished breastplate catching the dim torchlight, The burn scar creeping up his neck glows faintly in the flickering shadows. His hands grip the pommel of his greatsword, fingers tracing the worn groove of a prayer etched into the leather. His lips move in silent devotion, eyes scanning the room with the weight of past horrors. The air around him hums with quiet vigilance, a sentinel against unseen threats. "
"Broad-shouldered male paladin in tarnished silver breastplate, standing guard beneath a stone arch. Burn scar on neck, short-cropped hair, skull dent visible. Gripping a greatsword point-down, hands on pommel, lips moving in silent prayer. Torchlight flickering, dim medieval corridor, shadows stretching. Sweat-darkened gambeson, solemn expression, atmosphere of vigilance, fantasy setting."

prompt = "tavern scenes, high quality, fantasy tavern scene, tavern atmosphere, warm lights, wide shot, wide-angle lens, super vista, super wide Angle, man, paladin Ser Ryam, standing, both hands gripping pommel of sword, tarnished breastplate, burn scar on neck, inside tavern, full body, background characters, Masterpiece, newest, perfect quality"
negative_prompt = "EasyNegative, portrait, ugly, deformed, noisy, blurry, distorted, out of focus, bad anatomy, extra limbs, poorly drawn face, poorly drawn hands, missing fingers, bad quality, medium quality"

generate_image(
    pipe=pipe,
    prompt=prompt,
    negative_prompt=negative_prompt,
    steps=55,
    cfg_scale=7,
    seed=2903712067,
)