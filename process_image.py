from torchvision import transforms
from PIL import Image
import os

# ========== 1. 基础配置（与你的原有代码对齐） ==========
sample_size = [1920, 1080]  # 模型目标尺寸：高1280 × 宽704
target_height, target_width = sample_size[0], sample_size[1]

# ========== 2. 定义「保持比例 + 中心裁剪」转换器 ==========
transform = transforms.Compose([
    # 第一步：按目标高度缩放（保持宽高比），最短边达标
    transforms.Resize(
        size=target_height,
        interpolation=transforms.InterpolationMode.BILINEAR  # 平衡速度和质量
    ),
    # 第二步：中心裁剪到目标尺寸（切除多余部分，不拉伸）
    transforms.CenterCrop(size=(target_height, target_width))
])

# ========== 3. 读取+处理图片 ==========
original_image_path = "style3d.png"  # 你的原始参考图片路径
processed_image_path = "style3d_processed.png"  # 处理后图片的保存路径

# 读取图片（确保兼容不同格式，convert("RGB") 避免透明通道问题）
original_image = Image.open(original_image_path).convert("RGB")

# 应用处理（无拉伸，保持人物比例）
processed_image = transform(original_image)

# ========== 4. 保存处理后的图片 ==========
# 确保保存目录存在（若路径包含子目录，如 "assets/processed/xxx.jpg"）
save_dir = os.path.dirname(processed_image_path)
if save_dir and not os.path.exists(save_dir):
    os.makedirs(save_dir, exist_ok=True)

# 保存图片（支持 JPG/PNG 等格式，根据后缀自动识别）
processed_image.save(processed_image_path, quality=95)  # quality=95 保留高清细节（JPG格式有效）

print(f"图片处理完成！已保存到：{os.path.abspath(processed_image_path)}")
print(f"处理后尺寸：{processed_image.size}（宽×高）")  # 输出：(704, 1280)，与模型要求匹配
