import torch
from flux_pipeline_hiflow import FluxPipeline
from transformer_flux import FluxTransformer2DModel
from utils import set_seeds
import pdb

seed = 3407
device = "cuda"
model_path = "black-forest-labs/FLUX.1-dev"

transformer = FluxTransformer2DModel.from_pretrained(model_path, subfolder="transformer", torch_dtype=torch.float16)
pipe = FluxPipeline.from_pretrained(model_path, transformer=None,  torch_dtype=torch.float16)
pipe.transformer = transformer
pipe.scheduler.use_dynamic_shifting = False
pipe.to(device)

set_seeds(seed)

prompt = "A robot standing in the rain reading newspaper, rusty and worn down, in a dystopian cyberpunk street, photo-realistic, urbanpunk."

images = pipe(
    prompt = prompt,
    # --------- Default Inference Parameters for Flux-dev 1K generation -----------
    height = 1024,
    width = 1024,
    guidance_scale = 3.5,
    num_inference_steps = 30,
    max_sequence_length = 512,
    # -------- Flux High Resolution Inference Toolkits ----
    ntk_factor = [10.0, 10.0,], 
    proportional_attention = True, 
    text_duplication = False, 
    swin_pachify = True, 
    # --------------- HiFlow Parameters ---------
    target_heights = [2048, 4096], 
    target_widths = [2048, 4096], 
    num_inference_steps_highres = [16, 10,], 
    filter_ratio = [0.2, 0.2,], 
    guidance_scale_highres = [4.5, 6,], 
    structure_guidance = "fft", # ["fft", "dwt"]
    alphas = [1.0, 0.25,], 
    betas = [0.5, 0.5,],
    upsampling_choice = "latent", # ["latent", "pixel"]
    flow_choice = "v_theta",
    generator=torch.Generator("cuda").manual_seed(seed),
    )