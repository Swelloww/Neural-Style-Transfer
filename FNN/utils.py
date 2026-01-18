import torch
from PIL import Image
from torchvision import transforms

def load_image(filename, size=None, scale=None):
    img = Image.open(filename).convert('RGB')
    if size is not None:
        # 替换此处：ANTIALIAS -> LANCZOS
        img = img.resize((size, size), Image.LANCZOS) 
    elif scale is not None:
        # 替换此处：ANTIALIAS -> LANCZOS
        img = img.resize((int(img.size[0] / scale), int(img.size[1] / scale)), Image.LANCZOS)
    return img

def save_image(filename, data):
    img = data.clone().clamp(0, 255).numpy()
    img = img.transpose(1, 2, 0).astype("uint8")
    img = Image.fromarray(img)
    img.save(filename)

def normalize_batch(batch):
    # VGG 需要的归一化参数
    mean = batch.new_tensor([0.485, 0.456, 0.406]).view(-1, 1, 1)
    std = batch.new_tensor([0.229, 0.224, 0.225]).view(-1, 1, 1)
    return (batch.div(255.0) - mean) / std

# === 核心算法：Sliced Wasserstein Loss ===
def sliced_wasserstein_loss(input_features, target_features, num_projections=64):
    """
    计算两个特征图分布之间的 Sliced Wasserstein 距离
    """
    # [B, C, H, W] -> [B, C, H*W]
    b, c, h, w = input_features.shape
    input_flat = input_features.view(b, c, -1)
    target_flat = target_features.view(b, c, -1)
    
    # 生成随机投影向量 [C, num_projections]
    # 为了稳定，可以在此处固定随机种子，或者每次迭代随机
    device = input_features.device
    projections = torch.randn(c, num_projections).to(device)
    projections = projections / torch.sqrt(torch.sum(projections**2, dim=0, keepdim=True))
    
    # 投影: [B, num_projections, H*W]
    input_projs = torch.bmm(projections.transpose(0, 1).unsqueeze(0).expand(b, -1, -1), input_flat)
    target_projs = torch.bmm(projections.transpose(0, 1).unsqueeze(0).expand(b, -1, -1), target_flat)
    
    # 排序 (Wasserstein Distance 的一维闭式解需要排序)
    input_projs, _ = torch.sort(input_projs, dim=-1)
    target_projs, _ = torch.sort(target_projs, dim=-1)
    
    # 计算 L2 损失
    return torch.mean((input_projs - target_projs)**2)