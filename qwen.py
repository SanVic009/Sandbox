import warnings
import resource
import torch
from diffusers import DiffusionPipeline
from datetime import datetime

# --- Suppress all warnings ---
warnings.filterwarnings("ignore")

# --- Record start time ---
start_time = datetime.now()

# --- Limit max RAM to 8GB ---
GB = 8
limit_bytes = GB * 1024 ** 3
resource.setrlimit(resource.RLIMIT_AS, (limit_bytes, limit_bytes))

# --- Setup model and device ---
model_name = "Qwen/Qwen-Image"

if torch.cuda.is_available():
    torch_dtype = torch.float16
    device = "cuda"
    generator = torch.Generator(device=device).manual_seed(42)
else:
    torch_dtype = torch.float32
    device = "cpu"
    generator = torch.Generator().manual_seed(42)

# --- Load pipeline ---
pipe = DiffusionPipeline.from_pretrained(
    model_name,
    torch_dtype=torch_dtype,
    variant="fp16" if torch_dtype == torch.float16 else None
)
pipe.to(device)

# --- Enable memory-efficient features ---
pipe.enable_attention_slicing()
pipe.enable_model_cpu_offload()
pipe.enable_vae_slicing()
# pipe.safety_checker = lambda images, **kwargs: (images, False)  # Optional

# --- Prompt ---
positive_magic = {
    "en": " Ultra HD, 4K, cinematic composition.",
    "zh": " 超清，4K，电影级构图"
}

prompt = (
    '''A coffee shop entrance features a chalkboard sign reading "Qwen Coffee 😊 $2 per cup," '''
    '''with a neon light beside it displaying "通义千问". Next to it hangs a poster showing a beautiful '''
    '''Chinese woman, and beneath the poster is written "π≈3.1415926-53589793-23846264-33832795-02384197".'''
    + positive_magic["en"]
)

negative_prompt = " "

# --- Aspect Ratio 16:9 ---
width, height = 1664, 928

# --- Generate image ---
image = pipe(
    prompt=prompt,
    negative_prompt=negative_prompt,
    width=width,
    height=height,
    num_inference_steps=30,
    true_cfg_scale=4.0,
    generator=generator
).images[0]

# --- Save image ---
image.save("example.png")

# --- Record end time and print summary ---
end_time = datetime.now()
print(f"Started at: {start_time}")
print(f"Ended at:   {end_time}")
print(f"Total time: {end_time - start_time}")
