import time
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from models import TransformerNet, Vgg19
from utils import load_image, save_image,sliced_wasserstein_loss, normalize_batch
import os

# 配置参数
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
TRAIN_IMAGE_SIZE = 256
DATASET_PATH = "./coco_data" # 图片文件夹
STYLE_IMAGE_PATH = "./style_sunrise.jpg" # 风格图
BATCH_SIZE = 4
EPOCHS = 200
LR = 1e-3
CONTENT_WEIGHT = 1e5
STYLE_WEIGHT = 1e7 # 风格权重
TV_WEIGHT = 1e-6    # 全变分损失
SAVE_MODEL_PATH = "style_swd_checkpoint.pth" # 定义统一的保存路径

OUTPUT_DIR = "outputs/training_samples" # 预览图保存文件夹
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

SAMPLE_CONTENT_IMAGE = "content_image/boy.jpg"

def train():
    
    # 数据准备
    
    transform = transforms.Compose([
        transforms.Resize(TRAIN_IMAGE_SIZE),
        transforms.CenterCrop(TRAIN_IMAGE_SIZE),
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x.mul(255))
    ])
    train_dataset = datasets.ImageFolder(DATASET_PATH, transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

    # 模型初始化
    transformer = TransformerNet().to(DEVICE)
    vgg = Vgg19(requires_grad=False).to(DEVICE)
    optimizer = optim.Adam(transformer.parameters(), lr=LR)


    start_epoch = 0
    if os.path.exists(SAVE_MODEL_PATH):
        print(f"检测到保存的检查点，正在加载: {SAVE_MODEL_PATH}")
        checkpoint = torch.load(SAVE_MODEL_PATH)
        # 加载模型权重
        transformer.load_state_dict(checkpoint['model_state_dict'])
        # 加载优化器状态
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        # 读取中断时的 epoch
        start_epoch = checkpoint['epoch'] + 1
        print(f"加载成功！将从第 {start_epoch + 1} 个 Epoch 继续训练。")


    # 预计算风格特征
    print(f"Pre-calculating features for style: {STYLE_IMAGE_PATH}")
    style_img = load_image(STYLE_IMAGE_PATH, size=TRAIN_IMAGE_SIZE)
    style_tensor = transforms.ToTensor()(style_img).unsqueeze(0).to(DEVICE).mul(255)
    
    with torch.no_grad():
        style_norm = normalize_batch(style_tensor)
        # 提取风格图的四层特征
        style_features = vgg(style_norm)
        
        style_features = [f.repeat(BATCH_SIZE, 1, 1, 1) for f in style_features]
 
    print(f"Total batches to train: {len(train_loader)}")
    sample_img = load_image(SAMPLE_CONTENT_IMAGE, size=TRAIN_IMAGE_SIZE) if os.path.exists(SAMPLE_CONTENT_IMAGE) else None


    # 训练循环
    for epoch in range(EPOCHS):
        transformer.train()
        for batch_id, (x, _) in enumerate(train_loader):
            n_batch = len(x)
            x = x.to(DEVICE)
            optimizer.zero_grad()

            # 前向传播：生成图片
            y = transformer(x)
            
            # 归一化输入以便 VGG 提取特征
            x_norm = normalize_batch(x)
            y_norm = normalize_batch(y)

            # 提取特征
            x_features = vgg(x_norm)
            y_features = vgg(y_norm)

            # 损失计算
            # 内容损失
            content_loss = CONTENT_WEIGHT * torch.nn.functional.mse_loss(y_features[1], x_features[1])

            # 风格损失（使用SWD）
            style_loss = 0.
            for ft_y, ft_style in zip(y_features, style_features[:n_batch]): 
                # 处理最后一个 batch 可能不足 batch_size 的情况
                curr_batch_style = ft_style[:n_batch] 
                style_loss += sliced_wasserstein_loss(ft_y, curr_batch_style)
            style_loss *= STYLE_WEIGHT

            # TV Loss
            tv_loss = TV_WEIGHT * (torch.sum(torch.abs(y[:, :, :, :-1] - y[:, :, :, 1:])) + 
                                   torch.sum(torch.abs(y[:, :, :-1, :] - y[:, :, 1:, :])))

            total_loss = content_loss + style_loss + tv_loss
            total_loss.backward()
            optimizer.step()

            if (batch_id + 1) % 100 == 0:
                print(f"Epoch {epoch+1}, Batch {batch_id+1}: Content: {content_loss.item():.2f}, Style(SWD): {style_loss.item():.2f}")
        
        
        transformer.eval() # 切换到评价模式
        with torch.no_grad():
            # 如果没指定预览图，就用当前 Batch 的最后一张图
            test_x = x[0:1] 
            if sample_img is not None:
                test_x = transforms.ToTensor()(sample_img).unsqueeze(0).to(DEVICE).mul(255)
            
            output = transformer(test_x)
            sample_filename = os.path.join(OUTPUT_DIR, f"epoch_{epoch+1}_sample.png")
            save_image(sample_filename, output[0].cpu())
            print(f"--> Sample image saved: {sample_filename}")
        
        state = {
            'epoch': epoch,
            'model_state_dict': transformer.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
        }
        torch.save(state, SAVE_MODEL_PATH)
        print(f"Epoch {epoch+1} completed，checkpoint has been saved.")
    # 保存集成好的权重
    save_name = "style_swd_integrated.pth"
    torch.save(transformer.state_dict(), save_name)
    print(f"Model saved to {save_name}")

if __name__ == "__main__":
    train()