import os
import sys

import numpy as np
import torch
from diffusers import FlowMatchEulerDiscreteScheduler
from omegaconf import OmegaConf
from PIL import Image
from transformers import AutoTokenizer

current_file_path = os.path.abspath(__file__)
project_roots = [os.path.dirname(current_file_path), os.path.dirname(os.path.dirname(current_file_path)), os.path.dirname(os.path.dirname(os.path.dirname(current_file_path)))]
for project_root in project_roots:
    sys.path.insert(0, project_root) if project_root not in sys.path else None

from videox_fun.dist import set_multi_gpus_devices, shard_model
from videox_fun.models import (AutoencoderKLWan, AutoencoderKLWan3_8, AutoTokenizer, CLIPModel,
                               WanT5EncoderModel, Wan2_2Transformer3DModel)
from videox_fun.data.dataset_image_video import process_pose_file
from videox_fun.models.cache_utils import get_teacache_coefficients
from videox_fun.pipeline import Wan2_2FunControlPipeline, WanPipeline
from videox_fun.utils.fp8_optimization import (convert_model_weight_to_float8,
                                               convert_weight_dtype_wrapper,
                                               replace_parameters_by_name)
from videox_fun.utils.lora_utils import merge_lora, unmerge_lora
from videox_fun.utils.utils import (filter_kwargs, get_image_latent, get_image_to_video_latent,
                                    get_video_to_video_latent,
                                    save_videos_grid)
from videox_fun.utils.fm_solvers import FlowDPMSolverMultistepScheduler
from videox_fun.utils.fm_solvers_unipc import FlowUniPCMultistepScheduler
from PIL import Image, ImageOps
import tempfile
import cv2
def letterbox_pil(img: Image.Image, target_size):
    """
    保持宽高比缩放并在两侧/上下填充黑边，使结果正好为 target_size (H, W)
    target_size: [H, W]
    返回 PIL.Image (RGB)
    """
    target_h, target_w = target_size[0], target_size[1]
    img = img.convert("RGB")
    orig_w, orig_h = img.size[0], img.size[1]
    # 计算缩放
    scale = min(target_w / orig_w, target_h / orig_h)
    new_w, new_h = int(orig_w * scale), int(orig_h * scale)
    resized = img.resize((new_w, new_h), Image.LANCZOS)
    # pad to target
    pad_left = (target_w - new_w) // 2
    pad_top  = (target_h - new_h) // 2
    pad_right = target_w - new_w - pad_left
    pad_bottom = target_h - new_h - pad_top
    padded = ImageOps.expand(resized, border=(pad_left, pad_top, pad_right, pad_bottom), fill=(0,0,0))
    return padded

def preprocess_ref_image_for_model(ref_image_path, target_size):
    """
    把参考图片做 letterbox，然后写成临时文件并返回临时路径。
    如果 ref_image_path 本身是 None，返回 None。
    """
    if ref_image_path is None:
        return None
    im = Image.open(ref_image_path)
    im2 = letterbox_pil(im, target_size)
    tmp = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
    im2.save(tmp.name)
    return tmp.name

def preprocess_video_letterbox(in_video_path, target_size, out_video_path=None, target_fps=None):
    """
    把输入视频的每帧做 letterbox（保持宽高比 + 填充），写入 out_video_path（如果 None 则生成临时文件）
    target_size: [H, W] (与 get_*_latent 使用的 sample_size 一致)
    返回输出视频路径
    """
    if in_video_path is None:
        return None
    cap = cv2.VideoCapture(in_video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {in_video_path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or (target_fps or 24)
    if target_fps is not None:
        fps = target_fps

    out_tmp = out_video_path
    if out_tmp is None:
        out_tmp = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False).name

    # use mp4v codec for portability
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    target_w = target_size[1]
    target_h = target_size[0]
    writer = cv2.VideoWriter(out_tmp, fourcc, fps, (target_w, target_h))

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        # frame: HxW BGR
        h, w = frame.shape[:2]
        scale = min(target_w / w, target_h / h)
        new_w, new_h = int(w * scale), int(h * scale)
        resized = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA)
        pad_left = (target_w - new_w) // 2
        pad_top  = (target_h - new_h) // 2
        pad_right = target_w - new_w - pad_left
        pad_bottom = target_h - new_h - pad_top
        padded = cv2.copyMakeBorder(resized, pad_top, pad_bottom, pad_left, pad_right, borderType=cv2.BORDER_CONSTANT, value=(0,0,0))
        writer.write(padded)
    cap.release()
    writer.release()
    return out_tmp
# GPU memory mode, which can be chosen in [model_full_load, model_cpu_offload_and_qfloat8, model_cpu_offload, model_cpu_offload_and_qfloat8, sequential_cpu_offload].
# model_full_load means that the entire model will be moved to the GPU.
# 
# model_full_load_and_qfloat8 means that the entire model will be moved to the GPU,
# and the transformer model has been quantized to float8, which can save more GPU memory. 
# 
# model_cpu_offload means that the entire model will be moved to the CPU after use, which can save some GPU memory.
# 
# model_cpu_offload_and_qfloat8 indicates that the entire model will be moved to the CPU after use, 
# and the transformer model has been quantized to float8, which can save more GPU memory. 
# 
# sequential_cpu_offload means that each layer of the model will be moved to the CPU after use, 
# resulting in slower speeds but saving a large amount of GPU memory.
GPU_memory_mode     = "sequential_cpu_offload"
# Multi GPUs config
# Please ensure that the product of ulysses_degree and ring_degree equals the number of GPUs used. 
# For example, if you are using 8 GPUs, you can set ulysses_degree = 2 and ring_degree = 4.
# If you are using 1 GPU, you can set ulysses_degree = 1 and ring_degree = 1.
ulysses_degree      = 1
ring_degree         = 1
# Use FSDP to save more GPU memory in multi gpus.
fsdp_dit            =True
fsdp_text_encoder   = True
# Compile will give a speedup in fixed resolution and need a little GPU memory. 
# The compile_dit is not compatible with the fsdp_dit and sequential_cpu_offload.
compile_dit         = False

# Support TeaCache.
enable_teacache     = True
# Recommended to be set between 0.05 and 0.30. A larger threshold can cache more steps, speeding up the inference process, 
# but it may cause slight differences between the generated content and the original content.
# # --------------------------------------------------------------------------------------------------- #
# | Model Name          | threshold | Model Name          | threshold |
# | Wan2.2-T2V-A14B     | 0.10~0.15 | Wan2.2-I2V-A14B     | 0.15~0.20 |
# | Wan2.2-Fun-A14B-*   | 0.15~0.20 |
# # --------------------------------------------------------------------------------------------------- #
teacache_threshold  = 0.10
# The number of steps to skip TeaCache at the beginning of the inference process, which can
# reduce the impact of TeaCache on generated video quality.
num_skip_start_steps = 5
# Whether to offload TeaCache tensors to cpu to save a little bit of GPU memory.
teacache_offload    = False

# Skip some cfg steps in inference
# Recommended to be set between 0.00 and 0.25
cfg_skip_ratio      = 0

# Riflex config
enable_riflex       = False
# Index of intrinsic frequency
riflex_k            = 6

# Config and model path
config_path         = "config/wan2.2/wan_civitai_5b.yaml"
# model path
model_name          = "../models/Wan2.2-Fun-5B-Control/"

# Choose the sampler in "Flow", "Flow_Unipc", "Flow_DPM++"
sampler_name        = "Flow"
# [NOTE]: Noise schedule shift parameter. Affects temporal dynamics. 
# Used when the sampler is in "Flow_Unipc", "Flow_DPM++".
shift               = 5

# Load pretrained model if need
# The transformer_path is used for low noise model, the transformer_high_path is used for high noise model.
# Since Wan2.2-5b consists of only one model, only transformer_path is used.
transformer_path        = None
transformer_high_path   = None
vae_path                = None
# Load lora model if need
# The lora_path is used for low noise model, the lora_high_path is used for high noise model.
# Since Wan2.2-5b consists of only one model, only lora_path is used.
lora_path               = None
lora_high_path          = None

# Other params
sample_size         = [704, 1280]
video_length        = 49
fps                 = 8

# Use torch.float16 if GPU does not support torch.bfloat16
# ome graphics cards, such as v100, 2080ti, do not support torch.bfloat16
weight_dtype            = torch.bfloat16
control_video           = "asset/control_video.mp4"
control_camera_txt      = None
start_image             = None
end_image               = None
ref_image               = "asset/processed_girl.png"

# 使用更长的neg prompt如"模糊，突变，变形，失真，画面暗，文本字幕，画面固定，连环画，漫画，线稿，没有主体。"，可以增加稳定性
# 在neg prompt中添加"安静，固定"等词语可以增加动态性。
#prompt              = "一位年轻女子站在阳光明媚的海岸线上，身穿深蓝色背心与清爽的白色衬衫，外搭一条简洁的白色围裙，围裙在轻拂的海风中微微飘动。她拥有一头鲜艳的紫色长发，在风中轻盈舞动，发间系着一个精致的黑色蝴蝶结，与身后柔和的蔚蓝天空形成鲜明对比。她面容清秀，眉目精致，透着一股甜美的青春气息；神情柔和，略带羞涩，目光静静地凝望着远方的地平线，双手自然交叠于身前，仿佛沉浸在思绪之中。在她身后，是辽阔无垠、波光粼粼的大海，阳光洒在海面上，映出温暖的金色光晕。"

prompt              = "一位年轻女子有一头长直发，发色为深棕色或接近黑色，发丝在微风中轻轻飞扬，呈现自然动态感。她的皮肤白皙，五官精致，眼睛大而明亮，眉毛修饰得自然，嘴唇微微上扬，表情柔和而略带温暖。她穿着一件白色无袖上衣，上衣带有轻薄的褶皱装饰，整体风格轻盈、夏日感强。下半身穿白色超短裙，整体衣着干净简约。她双臂交叉放在胸前，身体稍微向一侧倾斜，姿态自然、优雅。目光直视镜头，神情平静而专注，带有一丝柔美感。背景为绿色的自然环境（可能是树木或草地），呈现虚化效果（浅景深），使人物更加突出。光线柔和，似乎是自然光，光线从左侧或上方照射过来，照亮她的脸部和头发，形成微微的光晕效果，增加画面梦幻感。整张图片色调明亮柔和，强调清新、纯净的气质，像是夏日清晨或傍晚的自然光下拍摄的肖像，带有轻盈、唯美的艺术感。"
#prompt="一位年轻的欧美女性有着一头棕红色的长卷发，头发披散着，精致的妆容，眼妆深邃，佩戴着黑色耳环。她穿着一件无袖的碎花连衣裙，底色为浅灰色，上面点缀着红、白、黄等颜色的花朵图案，裙子为方形领口，腰部收束，裙摆呈伞状展开，风格清新又带有几分俏皮。她脚上穿着一双白色高跟鞋，鞋头为圆形，后跟处有黑色蝴蝶结装饰，增添了甜美元素。她的姿势是一只手叉腰，另一只手自然下垂，双腿交叉站立，姿态优雅且富有表现力。背景是纯灰色的，简洁干净，突出了人物的形象。整体画面展现出一种时尚、甜美的风格，人物的穿搭和姿态都传递出自信的气质。"
negative_prompt     = "色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走"

# Using longer neg prompt such as "Blurring, mutation, deformation, distortion, dark and solid, comics, text subtitles, line art." can increase stability
# Adding words such as "quiet, solid" to the neg prompt can increase dynamism.
# prompt                  = "A young woman with beautiful, clear eyes and blonde hair stands in the forest, wearing a white dress and a crown. Her expression is serene, reminiscent of a movie star, with fair and youthful skin. Her brown long hair flows in the wind. The video quality is very high, with a clear view. High quality, masterpiece, best quality, high resolution, ultra-fine, fantastical."
# negative_prompt         = "Twisted body, limb deformities, text captions, comic, static, ugly, error, messy code."
guidance_scale          = 7.0
seed                    = 43
num_inference_steps     = 40
# The lora_weight is used for low noise model, the lora_high_weight is used for high noise model.
lora_weight             = 0.55
lora_high_weight        = 0.55
save_path               = "samples/wan-videos-fun-control"

#device = set_multi_gpus_devices(ulysses_degree, ring_degree)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
config = OmegaConf.load(config_path)
boundary = config['transformer_additional_kwargs'].get('boundary', 0.875)

transformer = Wan2_2Transformer3DModel.from_pretrained(
    os.path.join(model_name, config['transformer_additional_kwargs'].get('transformer_low_noise_model_subpath', 'transformer')),
    transformer_additional_kwargs=OmegaConf.to_container(config['transformer_additional_kwargs']),
    low_cpu_mem_usage=True,
    torch_dtype=weight_dtype,
)
if config['transformer_additional_kwargs'].get('transformer_combination_type', 'single') == "moe":
    transformer_2 = Wan2_2Transformer3DModel.from_pretrained(
        os.path.join(model_name, config['transformer_additional_kwargs'].get('transformer_high_noise_model_subpath', 'transformer')),
        transformer_additional_kwargs=OmegaConf.to_container(config['transformer_additional_kwargs']),
        low_cpu_mem_usage=True,
        torch_dtype=weight_dtype,
    )
else:
    transformer_2 = None

if transformer_path is not None:
    print(f"From checkpoint: {transformer_path}")
    if transformer_path.endswith("safetensors"):
        from safetensors.torch import load_file, safe_open
        state_dict = load_file(transformer_path)
    else:
        state_dict = torch.load(transformer_path, map_location="cpu")
    state_dict = state_dict["state_dict"] if "state_dict" in state_dict else state_dict

    m, u = transformer.load_state_dict(state_dict, strict=False)
    print(f"missing keys: {len(m)}, unexpected keys: {len(u)}")

if transformer_2 is not None:
    if transformer_high_path is not None:
        print(f"From checkpoint: {transformer_high_path}")
        if transformer_high_path.endswith("safetensors"):
            from safetensors.torch import load_file, safe_open
            state_dict = load_file(transformer_high_path)
        else:
            state_dict = torch.load(transformer_high_path, map_location="cpu")
        state_dict = state_dict["state_dict"] if "state_dict" in state_dict else state_dict

        m, u = transformer_2.load_state_dict(state_dict, strict=False)
        print(f"missing keys: {len(m)}, unexpected keys: {len(u)}")

# Get Vae
Chosen_AutoencoderKL = {
    "AutoencoderKLWan": AutoencoderKLWan,
    "AutoencoderKLWan3_8": AutoencoderKLWan3_8
}[config['vae_kwargs'].get('vae_type', 'AutoencoderKLWan')]
vae = Chosen_AutoencoderKL.from_pretrained(
    os.path.join(model_name, config['vae_kwargs'].get('vae_subpath', 'vae')),
    additional_kwargs=OmegaConf.to_container(config['vae_kwargs']),
).to(weight_dtype)

if vae_path is not None:
    print(f"From checkpoint: {vae_path}")
    if vae_path.endswith("safetensors"):
        from safetensors.torch import load_file, safe_open
        state_dict = load_file(vae_path)
    else:
        state_dict = torch.load(vae_path, map_location="cpu")
    state_dict = state_dict["state_dict"] if "state_dict" in state_dict else state_dict

    m, u = vae.load_state_dict(state_dict, strict=False)
    print(f"missing keys: {len(m)}, unexpected keys: {len(u)}")

# Get Tokenizer
tokenizer = AutoTokenizer.from_pretrained(
    os.path.join(model_name, config['text_encoder_kwargs'].get('tokenizer_subpath', 'tokenizer')),
)

# Get Text encoder
text_encoder = WanT5EncoderModel.from_pretrained(
    os.path.join(model_name, config['text_encoder_kwargs'].get('text_encoder_subpath', 'text_encoder')),
    additional_kwargs=OmegaConf.to_container(config['text_encoder_kwargs']),
    low_cpu_mem_usage=True,
    torch_dtype=weight_dtype,
)
text_encoder = text_encoder.eval()

# Get Scheduler
Chosen_Scheduler = scheduler_dict = {
    "Flow": FlowMatchEulerDiscreteScheduler,
    "Flow_Unipc": FlowUniPCMultistepScheduler,
    "Flow_DPM++": FlowDPMSolverMultistepScheduler,
}[sampler_name]
if sampler_name == "Flow_Unipc" or sampler_name == "Flow_DPM++":
    config['scheduler_kwargs']['shift'] = 1
scheduler = Chosen_Scheduler(
    **filter_kwargs(Chosen_Scheduler, OmegaConf.to_container(config['scheduler_kwargs']))
)

# Get Pipeline
pipeline = Wan2_2FunControlPipeline(
    transformer=transformer,
    transformer_2=transformer_2,
    vae=vae,
    tokenizer=tokenizer,
    text_encoder=text_encoder,
    scheduler=scheduler,
)
if ulysses_degree > 1 or ring_degree > 1:
    from functools import partial
    transformer.enable_multi_gpus_inference()
    if transformer_2 is not None:
        transformer_2.enable_multi_gpus_inference()
    if fsdp_dit:
        shard_fn = partial(shard_model, device_id=device, param_dtype=weight_dtype)
        pipeline.transformer = shard_fn(pipeline.transformer)
        if transformer_2 is not None:
            pipeline.transformer_2 = shard_fn(pipeline.transformer_2)
        print("Add FSDP DIT")
    if fsdp_text_encoder:
        shard_fn = partial(shard_model, device_id=device, param_dtype=weight_dtype)
        pipeline.text_encoder = shard_fn(pipeline.text_encoder)
        print("Add FSDP TEXT ENCODER")

if compile_dit:
    for i in range(len(pipeline.transformer.blocks)):
        pipeline.transformer.blocks[i] = torch.compile(pipeline.transformer.blocks[i])
    if transformer_2 is not None:
        for i in range(len(pipeline.transformer_2.blocks)):
            pipeline.transformer_2.blocks[i] = torch.compile(pipeline.transformer_2.blocks[i])
    print("Add Compile")

if GPU_memory_mode == "sequential_cpu_offload":
    replace_parameters_by_name(transformer, ["modulation",], device=device)
    transformer.freqs = transformer.freqs.to(device=device)
    if transformer_2 is not None:
        replace_parameters_by_name(transformer_2, ["modulation",], device=device)
        transformer_2.freqs = transformer_2.freqs.to(device=device)
    pipeline.enable_sequential_cpu_offload(device=device)
elif GPU_memory_mode == "model_cpu_offload_and_qfloat8":
    convert_model_weight_to_float8(transformer, exclude_module_name=["modulation",], device=device)
    convert_weight_dtype_wrapper(transformer, weight_dtype)
    if transformer_2 is not None:
        convert_model_weight_to_float8(transformer_2, exclude_module_name=["modulation",], device=device)
        convert_weight_dtype_wrapper(transformer_2, weight_dtype)
    pipeline.enable_model_cpu_offload(device=device)
elif GPU_memory_mode == "model_cpu_offload":
    pipeline.enable_model_cpu_offload(device=device)
elif GPU_memory_mode == "model_full_load_and_qfloat8":
    convert_model_weight_to_float8(transformer, exclude_module_name=["modulation",], device=device)
    convert_weight_dtype_wrapper(transformer, weight_dtype)
    if transformer_2 is not None:
        convert_model_weight_to_float8(transformer_2, exclude_module_name=["modulation",], device=device)
        convert_weight_dtype_wrapper(transformer_2, weight_dtype)
    pipeline.to(device=device)
else:
    pipeline.to(device=device)

coefficients = get_teacache_coefficients(model_name) if enable_teacache else None
if coefficients is not None:
    print(f"Enable TeaCache with threshold {teacache_threshold} and skip the first {num_skip_start_steps} steps.")
    pipeline.transformer.enable_teacache(
        coefficients, num_inference_steps, teacache_threshold, num_skip_start_steps=num_skip_start_steps, offload=teacache_offload
    )
    if transformer_2 is not None:
        pipeline.transformer_2.share_teacache(transformer=pipeline.transformer)

if cfg_skip_ratio is not None:
    print(f"Enable cfg_skip_ratio {cfg_skip_ratio}.")
    pipeline.transformer.enable_cfg_skip(cfg_skip_ratio, num_inference_steps)
    if transformer_2 is not None:
        pipeline.transformer_2.share_cfg_skip(transformer=pipeline.transformer)

generator = torch.Generator(device=device).manual_seed(seed)

if lora_path is not None:
    pipeline = merge_lora(pipeline, lora_path, lora_weight, device=device, dtype=weight_dtype)
    if transformer_2 is not None:
        pipeline = merge_lora(pipeline, lora_high_path, lora_high_weight, device=device, dtype=weight_dtype, sub_transformer_name="transformer_2")

with torch.no_grad():
    video_length = int((video_length - 1) // vae.config.temporal_compression_ratio * vae.config.temporal_compression_ratio) + 1 if video_length != 1 else 1
    latent_frames = (video_length - 1) // vae.config.temporal_compression_ratio + 1

    if enable_riflex:
        pipeline.transformer.enable_riflex(k = riflex_k, L_test = latent_frames)
        if transformer_2 is not None:
            pipeline.transformer_2.enable_riflex(k = riflex_k, L_test = latent_frames)
    
    inpaint_video, inpaint_video_mask, clip_image = get_image_to_video_latent(start_image, end_image, video_length=video_length, sample_size=sample_size)
    target_sample_size = sample_size  # 你的脚本里已有 sample_size = [1280, 704]

    ## 2) 如果有 ref_image 路径，把它处理为 letterboxed 临时文件并替换 ref_image 变量
    #if ref_image is not None:
    #    # 如果 ref_image 可能已经是 PIL.Image 或 tensor，你也可以先保存为临时文件，然后处理。
    #    # 这里假设 ref_image 是路径字符串（与你脚本中相同）
    #    tmp_ref_path = preprocess_ref_image_for_model(ref_image, target_sample_size)
    #    # 用临时处理后的路径替换 ref_image，这样后面 get_image_latent(tmp_ref_path, ...) 会得到正确尺寸
    #    ref_image = tmp_ref_path

    ## 3) 对 control_video 做 letterbox 生成临时视频并用它来生成 latent
    #if control_video is not None:
    #    tmp_control_video = preprocess_video_letterbox(control_video, target_sample_size, target_fps=fps)
    #    # 把要传给 get_video_to_video_latent 的路径换成临时生成的视频
    #    control_video_for = tmp_control_video
    #else:
    #    control_video = None
    if ref_image is not None:
        ref_image = get_image_latent(ref_image, sample_size=sample_size)
    
    if control_camera_txt is not None:
        input_video, input_video_mask = None, None
        control_camera_video = process_pose_file(control_camera_txt, sample_size[1], sample_size[0])
        control_camera_video = control_camera_video[:video_length].permute([3, 0, 1, 2]).unsqueeze(0)
    else:
        input_video, input_video_mask, _, _ = get_video_to_video_latent(control_video, video_length=video_length, sample_size=sample_size, fps=fps, ref_image=None)
        control_camera_video = None

    sample = pipeline(
        prompt, 
        num_frames = video_length,
        negative_prompt = negative_prompt,
        height      = sample_size[0],
        width       = sample_size[1],
        generator   = generator,
        guidance_scale = guidance_scale,
        num_inference_steps = num_inference_steps,

        video      = inpaint_video,
        mask_video   = inpaint_video_mask,
        control_video = input_video,
        control_camera_video = control_camera_video,
        ref_image = ref_image,
        boundary = boundary,
        shift = shift,
    ).videos

if lora_path is not None:
    pipeline = unmerge_lora(pipeline, lora_path, lora_weight, device=device, dtype=weight_dtype)
    if transformer_2 is not None:
        pipeline = unmerge_lora(pipeline, lora_high_path, lora_high_weight, device=device, dtype=weight_dtype, sub_transformer_name="transformer_2")

def save_results():
    if not os.path.exists(save_path):
        os.makedirs(save_path, exist_ok=True)

    index = len([path for path in os.listdir(save_path)]) + 1
    prefix = str(index).zfill(8)
    if video_length == 1:
        video_path = os.path.join(save_path, prefix + ".png")

        image = sample[0, :, 0]
        image = image.transpose(0, 1).transpose(1, 2)
        image = (image * 255).numpy().astype(np.uint8)
        image = Image.fromarray(image)
        image.save(video_path)
    else:
        video_path = os.path.join(save_path, prefix + ".mp4")
        save_videos_grid(sample, video_path, fps=fps)

if ulysses_degree * ring_degree > 1:
    import torch.distributed as dist
    if dist.get_rank() == 0:
        save_results()
else:
    save_results()
