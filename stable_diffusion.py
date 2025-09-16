from diffusers import StableDiffusionPipeline
import torch

# Use a real, working model - Stable Diffusion 1.5
model_name = "runwayml/stable-diffusion-v1-5"

# Check if CUDA is available
if torch.cuda.is_available():
    device = "cuda"
    torch_dtype = torch.float16
    print("Using device: CUDA")
else:
    device = "cpu"
    torch_dtype = torch.float32
    print("Using device: CPU")

# Load the pipeline with memory-efficient settings
pipe = StableDiffusionPipeline.from_pretrained(
    model_name,
    torch_dtype=torch_dtype,
    # low_cpu_mem_usage=True,          # ✅ reduces RAM usage
    safety_checker=None,             # ✅ saves memory
    requires_safety_checker=False
)

# Move to device and enable memory optimizations
pipe = pipe.to(device)
if device == "cuda":
    pipe.enable_attention_slicing()  # ✅ reduces VRAM usage

# Prompt setup
prompt = '''An artificial intelligence brain made of glowing circuits and data streams, hovering in a digital network space, futuristic interface, holographic elements, clean blue and white tones, high-resolution, minimal background, professional concept art.'''

positive_magic = "Ultra HD, 4K, cinematic composition."
negative_prompt = ""

# 16:9 resolution - reduced for better compatibility
width, height = 832, 464  # Smaller resolution to avoid memory issues

# Generate the image
image = pipe(
    prompt=prompt + " " + positive_magic,
    negative_prompt=negative_prompt,
    width=width,
    height=height,
    num_inference_steps=20,            # ✅ reduced for faster generation
    guidance_scale=7.5,
    generator=torch.Generator(device=device).manual_seed(42)
).images[0]

# Save the result
image.save("coffee_shop_output.png")
print("✅ Image generated and saved as 'coffee_shop_output.png'")