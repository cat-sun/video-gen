import os
import torch
import torch.distributed as dist
from diffusers import AutoencoderKLWan, LucyEditPipeline
from diffusers.utils import load_video, export_to_video

# 1. 初始化分布式环境 (USP 依赖 torch.distributed)
local_rank = int(os.environ.get("LOCAL_RANK", 0))
torch.cuda.set_device(local_rank)
device = f"cuda:{local_rank}"
def initialize_usp(device_type):
    import torch.distributed as dist
    from xfuser.core.distributed import initialize_model_parallel, init_distributed_environment
    if not dist.is_initialized():
        dist.init_process_group(backend="nccl", init_method="env://")
    init_distributed_environment(rank=dist.get_rank(), world_size=dist.get_world_size())
    initialize_model_parallel(
        sequence_parallel_degree=dist.get_world_size(),
        ring_degree=dist.get_world_size(), # 尝试增加 Ring 程度
    ulysses_degree=1,
    )
    torch.cuda.set_device(local_rank)
# 2. 启用 USP 初始化
use_usp = True # 开启开关
if use_usp:
   
    initialize_usp(device)
    # USP 会在这里接管通信域，实现长序列的横向切分

# 3. 加载模型 (注意：多卡模式下每个进程都会加载，但通过 USP 共享权重)
model_id = "/data/shared/models/Lucy-Edit-Dev"
vae = AutoencoderKLWan.from_pretrained(model_id, subfolder="vae", torch_dtype=torch.float32)

# 加载 Pipeline，不要用 device_map="auto"，由 USP 控制 device
pipe = LucyEditPipeline.from_pretrained(
    model_id, 
    vae=vae, 
    torch_dtype=torch.bfloat16
)
pipe.to(device)

# 4. 视频处理 (仅在主进程进行，或多卡同步)
url = "https://d2drjpuinn46lb.cloudfront.net/painter_original_edit.mp4"
video = load_video(url)[:81]
video = [v.resize((832, 480)) for v in video]

# 5. 推理
# 这里的 pipe 内部如果适配了 USP，会自动进行并行计算
with torch.no_grad():
    output = pipe(
        prompt="...",
        video=video,
        num_frames=81,
        height=480,
        width=832,
        guidance_scale=5.0
    ).frames[0]

# 6. 只有主进程保存视频
if local_rank == 0:
    export_to_video(output, "output_usp.mp4", fps=24)
if dist.is_initialized():
    dist.destroy_process_group()