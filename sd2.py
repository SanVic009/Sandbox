from diffusers import StableDiffusionPipeline
import torch

# Force GPU usage
if not torch.cuda.is_available():
    raise RuntimeError("CUDA is not available. This script requires a GPU.")

device = "cuda"
torch_dtype = torch.float16
print("✅ Using device: CUDA")

# Load the pipeline with memory-efficient settings
pipe = StableDiffusionPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5",
    torch_dtype=torch_dtype,
    safety_checker=None,
    requires_safety_checker=False,
)

# Move to GPU and enable memory optimizations
pipe = pipe.to(device)
pipe.enable_attention_slicing()

# Prompt setup
prompt = 'Elephants playing cricket on a sunny day. Realistic'
positive_magic = "Ultra HD, 4K, cinematic composition."
negative_prompt = ""

# 16:9 resolution
width, height = 600, 720

# Generate the image
image = pipe(
    prompt=prompt + " " + positive_magic,
    negative_prompt=negative_prompt,
    width=width,
    height=height,
    num_inference_steps=20,
    guidance_scale=7.5,
    generator=torch.Generator(device=device).manual_seed(42),
).images[0]

# Save the result
image.save("coffee_shop_output.png")
print("✅ Image generated and saved as 'coffee_shop_output.png'")