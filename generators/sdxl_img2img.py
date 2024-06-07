import torch
from diffusers import (
    StableDiffusionXLPipeline,
    UNet2DConditionModel,
    EulerDiscreteScheduler,
    AutoPipelineForImage2Image
)
from huggingface_hub import hf_hub_download
from diffusers.utils import load_image, make_image_grid
from safetensors.torch import load_file

device = "cuda" if torch.cuda.is_available() else "cpu"


class SDXLImg2ImgGenerator:
    def __init__(
        self,
        use_cpu_offload=False,
    ):
        
        base = "stabilityai/stable-diffusion-xl-base-1.0"
        repo = "ByteDance/SDXL-Lightning"
        ckpt = "sdxl_lightning_4step_unet.safetensors"

        unet = UNet2DConditionModel.from_config(base, subfolder="unet").to(
            device, torch.float16
        )
        unet.load_state_dict(load_file(hf_hub_download(repo, ckpt), device=device))
        
        lora_model_id ="JawadC/pecorino"
        self.pipe = AutoPipelineForImage2Image.from_pretrained(
        "stabilityai/stable-diffusion-xl-refiner-1.0", torch_dtype=torch.float16, variant="fp16", use_safetensors=True
        ).to(device)
        
        #self.pipe.load_lora_weights(lora_model_id)    
        self.pipe.enable_model_cpu_offload()
        self.pipe.scheduler = EulerDiscreteScheduler.from_config(
            self.pipe.scheduler.config, timestep_spacing="trailing"
        )
        self.pipe.set_progress_bar_config(disable=True)
        if use_cpu_offload:
            self.pipe.enable_sequential_cpu_offload()
        self.strength = 0.8

    def generate(self, prompt, image):
        images = self.pipe(prompt, image=image, strength = self.strength).images
        
        return images

