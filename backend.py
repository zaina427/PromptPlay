# @title Prepare Environment
!pip install torch==2.6.0 torchvision==0.21.0
%cd /content

!pip install -q torchsde einops diffusers accelerate xformers==0.0.29.post2
!pip install av
!git clone https://github.com/Isi-dev/ComfyUI
%cd /content/ComfyUI/custom_nodes
!git clone https://github.com/Isi-dev/ComfyUI_GGUF.git
%cd /content/ComfyUI/custom_nodes/ComfyUI_GGUF
!pip install -r requirements.txt
%cd /content/ComfyUI
!apt -y install -qq aria2 ffmpeg

useQ6 = False # @param {"type":"boolean"}

if useQ6:
    !aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/city96/Wan2.1-T2V-14B-gguf/resolve/main/wan2.1-t2v-14b-Q6_K.gguf -d /content/ComfyUI/models/unet -o wan2.1-t2v-14b-Q6_K.gguf
else:
    !aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/city96/Wan2.1-T2V-14B-gguf/resolve/main/wan2.1-t2v-14b-Q5_0.gguf -d /content/ComfyUI/models/unet -o wan2.1-t2v-14b-Q5_0.gguf

!aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/Comfy-Org/Wan_2.1_ComfyUI_repackaged/resolve/main/split_files/text_encoders/umt5_xxl_fp8_e4m3fn_scaled.safetensors -d /content/ComfyUI/models/text_encoders -o umt5_xxl_fp8_e4m3fn_scaled.safetensors
!aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/Comfy-Org/Wan_2.1_ComfyUI_repackaged/resolve/main/split_files/vae/wan_2.1_vae.safetensors -d /content/ComfyUI/models/vae -o wan_2.1_vae.safetensors

import torch
import numpy as np
from PIL import Image
import gc
import sys
import random
import os
import imageio
import subprocess
from google.colab import files
from IPython.display import display, HTML, Image as IPImage
sys.path.insert(0, '/content/ComfyUI')

from comfy import model_management

from nodes import (
    CheckpointLoaderSimple,
    CLIPLoader,
    CLIPTextEncode,
    VAEDecode,
    VAELoader,
    KSampler,
    UNETLoader
)

from custom_nodes.ComfyUI_GGUF.nodes import UnetLoaderGGUF
from comfy_extras.nodes_model_advanced import ModelSamplingSD3
from comfy_extras.nodes_hunyuan import EmptyHunyuanLatentVideo
from comfy_extras.nodes_images import SaveAnimatedWEBP
from comfy_extras.nodes_video import SaveWEBM

# unet_loader = UNETLoader()
unet_loader = UnetLoaderGGUF()
# model_sampling = ModelSamplingSD3()
clip_loader = CLIPLoader()
clip_encode_positive = CLIPTextEncode()
clip_encode_negative = CLIPTextEncode()
vae_loader = VAELoader()
empty_latent_video = EmptyHunyuanLatentVideo()
ksampler = KSampler()
vae_decode = VAEDecode()
save_webp = SaveAnimatedWEBP()
save_webm = SaveWEBM()

def clear_memory():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
    for obj in list(globals().values()):
        if torch.is_tensor(obj) or (hasattr(obj, "data") and torch.is_tensor(obj.data)):
            del obj
    gc.collect()

def save_as_mp4(images, filename_prefix, fps, output_dir="/content/ComfyUI/output"):
    os.makedirs(output_dir, exist_ok=True)
    output_path = f"{output_dir}/{filename_prefix}.mp4"

    frames = [(img.cpu().numpy() * 255).astype(np.uint8) for img in images]

    with imageio.get_writer(output_path, fps=fps) as writer:
        for frame in frames:
            writer.append_data(frame)

    return output_path

def save_as_webp(images, filename_prefix, fps, quality=90, lossless=False, method=4, output_dir="/content/ComfyUI/output"):
    """Save images as animated WEBP using imageio."""
    os.makedirs(output_dir, exist_ok=True)
    output_path = f"{output_dir}/{filename_prefix}.webp"


    frames = [(img.cpu().numpy() * 255).astype(np.uint8) for img in images]


    kwargs = {
        'fps': int(fps),
        'quality': int(quality),
        'lossless': bool(lossless),
        'method': int(method)
    }

    with imageio.get_writer(
        output_path,
        format='WEBP',
        mode='I',
        **kwargs
    ) as writer:
        for frame in frames:
            writer.append_data(frame)

    return output_path

def save_as_webm(images, filename_prefix, fps, codec="vp9", quality=32, output_dir="/content/ComfyUI/output"):
    """Save images as WEBM using imageio."""
    os.makedirs(output_dir, exist_ok=True)
    output_path = f"{output_dir}/{filename_prefix}.webm"


    frames = [(img.cpu().numpy() * 255).astype(np.uint8) for img in images]


    kwargs = {
        'fps': int(fps),
        'quality': int(quality),
        'codec': str(codec),
        'output_params': ['-crf', str(int(quality))]
    }

    with imageio.get_writer(
        output_path,
        format='FFMPEG',
        mode='I',
        **kwargs
    ) as writer:
        for frame in frames:
            writer.append_data(frame)

    return output_path

def save_as_image(image, filename_prefix, output_dir="/content/ComfyUI/output"):
    """Save single frame as PNG image."""
    os.makedirs(output_dir, exist_ok=True)
    output_path = f"{output_dir}/{filename_prefix}.png"

    frame = (image.cpu().numpy() * 255).astype(np.uint8)

    Image.fromarray(frame).save(output_path)

    return output_path

def generate_video(
    positive_prompt: str = "a fox moving quickly in a beautiful winter scenery nature trees mountains daytime tracking camera",
    negative_prompt: str = "色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走",
    width: int = 832,
    height: int = 480,
    seed: int = 82628696717253,
    steps: int = 30,
    cfg_scale: float = 1.0,
    sampler_name: str = "uni_pc",
    scheduler: str = "simple",
    frames: int = 33,
    fps: int = 16,
    output_format: str = "mp4"
):

    with torch.inference_mode():
        print("Loading Text_Encoder...")
        clip = clip_loader.load_clip("umt5_xxl_fp8_e4m3fn_scaled.safetensors", "wan", "default")[0]

        positive = clip_encode_positive.encode(clip, positive_prompt)[0]
        negative = clip_encode_negative.encode(clip, negative_prompt)[0]

        del clip
        torch.cuda.empty_cache()
        gc.collect()

        empty_latent = empty_latent_video.generate(width, height, frames, 1)[0]

        print("Loading Unet Model...")
        if useQ6:
            model = unet_loader.load_unet("wan2.1-t2v-14b-Q6_K.gguf")[0]
        else:
            model = unet_loader.load_unet("wan2.1-t2v-14b-Q5_0.gguf")[0]
        # model = model_sampling.patch(model, 8)[0]

        print("Generating video...")
        sampled = ksampler.sample(
            model=model,
            seed=seed,
            steps=steps,
            cfg=cfg_scale,
            sampler_name=sampler_name,
            scheduler=scheduler,
            positive=positive,
            negative=negative,
            latent_image=empty_latent
        )[0]

        del model
        torch.cuda.empty_cache()
        gc.collect()

        print("Loading VAE...")
        vae = vae_loader.load_vae("wan_2.1_vae.safetensors")[0]

        try:
            print("Decoding latents...")
            decoded = vae_decode.decode(vae, sampled)[0]

            del vae
            torch.cuda.empty_cache()
            gc.collect()

            output_path = ""
            if frames == 1:
                print("Single frame detected - saving as PNG image...")
                output_path = save_as_image(decoded[0], "ComfyUI")
                # print(f"Image saved as PNG: {output_path}")

                display(IPImage(filename=output_path))
            else:
                if output_format.lower() == "webm":
                    print("Saving as WEBM...")
                    output_path = save_as_webm(
                        decoded,
                        "ComfyUI",
                        fps=fps,
                        codec="vp9",
                        quality=10
                    )
                elif output_format.lower() == "mp4":
                    print("Saving as MP4...")
                    output_path = save_as_mp4(decoded, "ComfyUI", fps)
                else:
                    raise ValueError(f"Unsupported output format: {output_format}")

                # print(f"Video saved as {output_format.upper()}: {output_path}")

                display_video(output_path)

        except Exception as e:
            print(f"Error during decoding/saving: {str(e)}")
            raise
        finally:
            clear_memory()

def display_video(video_path):
    from IPython.display import HTML
    from base64 import b64encode

    video_data = open(video_path,'rb').read()

    # Determine MIME type based on file extension
    if video_path.lower().endswith('.mp4'):
        mime_type = "video/mp4"
    elif video_path.lower().endswith('.webm'):
        mime_type = "video/webm"
    elif video_path.lower().endswith('.webp'):
        mime_type = "image/webp"
    else:
        mime_type = "video/mp4"  # default

    data_url = f"data:{mime_type};base64," + b64encode(video_data).decode()

    display(HTML(f"""
    <video width=512 controls autoplay loop>
        <source src="{data_url}" type="{mime_type}">
    </video>
    """))

print("✅ Environment Setup Complete!")

