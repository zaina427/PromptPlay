# @title Generate Video/Image

positive_prompt = "lion walking in dessert" # @param {"type":"string"}
negative_prompt = "色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走" # @param {"type":"string"}
width = 832 # @param {"type":"number"}
height = 480 # @param {"type":"number"}
seed = 82628696717258 # @param {"type":"integer"}
steps = 20 # @param {"type":"integer", "min":1, "max":100}
cfg_scale = 3 # @param {"type":"number", "min":1, "max":20}
sampler_name = "uni_pc" # @param ["uni_pc", "euler", "dpmpp_2m", "ddim", "lms"]
scheduler = "simple" # @param ["simple", "normal", "karras", "exponential"]
frames = 24 # @param {"type":"integer", "min":1, "max":120}
fps = 16 # @param {"type":"integer", "min":1, "max":60}
output_format = "mp4" # @param ["mp4", "webm"]

# with torch.inference_mode():
generate_video(
    positive_prompt=positive_prompt,
    negative_prompt=negative_prompt,
    width=width,
    height=height,
    seed=seed,
    steps=steps,
    cfg_scale=cfg_scale,
    sampler_name=sampler_name,
    scheduler=scheduler,
    frames=frames,
    fps=fps,
    output_format=output_format
)
clear_memory()
