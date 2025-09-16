import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

from diffusers import DiffusionPipeline
import torch

if not torch.cuda.is_available():
    raise RuntimeError("CUDA is not available.")

device = "cuda"
torch_dtype = torch.float16
print("✅ Using device: CUDA")

pipe = DiffusionPipeline.from_pretrained(
    "stabilityai/sdxl-turbo",
    torch_dtype=torch_dtype,
    variant="fp16",
    low_cpu_mem_usage=True,
)

pipe.enable_model_cpu_offload()  # Offload during generation
pipe.enable_attention_slicing()

prompt = "Narendra Modi, Ultra HD, 4K, cinematic composition."
negative_prompt = ""
width, height = 384, 256  # 🔥 Minimal resolution for 3.7GB GPU

image = pipe(
    prompt=prompt,
    negative_prompt=negative_prompt,
    width=width,
    height=height,
    num_inference_steps=1,  # 🔥 Reduce step count
    guidance_scale=0.0,
    generator=torch.Generator(device=device).manual_seed(42),
).images[0]

image.save("modi_sdxl_turbo_lowres.png")
print("✅ Image generated and saved as 'modi_sdxl_turbo_lowres.png'")
