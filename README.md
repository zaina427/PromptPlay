# PromptPlay
Turn your imagination into animated video clips using this powerful AI pipeline based on Stable Diffusion + ComfyUI + GGUF-based UNet models.

Input: Text prompt
Output: High-quality videos (MP4, WEBM, or animated WEBP)

## Features

-Simple text-to-video generation
-Output in MP4 / WEBM / WEBP / PNG
-Uses Wan2.1-T2V-14B UNet in GGUF format (fast + efficient)
-UMT5-XXL text encoder for better language understanding
-Latent decoding with VAE for realistic frames
-Built on ComfyUI (modular, extendable)
-Optimized for Colab + GPU memory cleanup

## Folder Structure
/content/
├── ComfyUI/
│   ├── custom_nodes/
│   │   └── ComfyUI_GGUF/
│   ├── models/
│   │   ├── unet/
│   │   ├── vae/
│   │   └── text_encoders/
│   └── output/


## Workflow
### High-Level Pipeline
flowchart TD
    A[Text Prompt] --> B[Text Encoder (UMT5-XXL)]
    B --> C[Positive & Negative Embeddings]
    C --> D[Empty Latent Video Space]
    D --> E[UNet (Wan2.1-GGUF)]
    E --> F[Latent Sampled Frames]
    F --> G[VAE Decoder]
    G --> H[RGB Video Frames]
    H --> I[Save as MP4 / WEBM / WEBP / PNG]

## Internal Steps
-Load pre-trained models (UNet, VAE, CLIP/UMT5)
-Encode positive and negative prompts
-Generate an empty latent video grid
-Sample latent frames using UNet + KSampler
-Decode using VAE
-Save and visualize the final output

## Key Insights
-GGUF models enable efficient inference with lower memory use
-Using UMT5 for prompt encoding improves text understanding
-Built on ComfyUI, allowing modular node-based workflow
-The same model can produce single-frame images or full videos
-Video quality is strongly affected by:
-Prompt quality
-CFG scale
-Number of diffusion steps
-VAE and UNet resolution support

## Visualization
 ### Example Video Output (MP4)
Prompt: "A cyberpunk city at night with glowing neon lights, raining"
(or embed a real demo from your output folder or Hugging Face)

## How to Run
### Requirements

Google Colab with GPU (recommended)
Python 3.10+
PyTorch 2.6.0
ffmpeg, aria2

## Step-by-step (Colab)

Clone the repo and install dependencies

!git clone https://github.com/Isi-dev/ComfyUI
!git clone https://github.com/Isi-dev/ComfyUI_GGUF.git
!pip install torch==2.6.0 torchvision==0.21.0
!pip install -r requirements.txt
!apt install ffmpeg aria2

## Download Pretrained Models

### UNet
aria2c https://huggingface.co/.../wan2.1-t2v-14b-Q5_0.gguf

### Text Encoder
aria2c https://huggingface.co/.../umt5_xxl_fp8.safetensors

### VAE
aria2c https://huggingface.co/.../wan_2.1_vae.safetensors

## Run the Generator Function

generate_video(
    positive_prompt = "a dragon flying over mountains at sunset",
    negative_prompt = "blurry, low quality, jpeg artifacts",
    width = 832,
    height = 480,
    seed = 123456,
    steps = 30,
    cfg_scale = 1.5,
    sampler_name = "uni_pc",
    scheduler = "simple",
    frames = 32,
    fps = 16,
    output_format = "mp4"
)
