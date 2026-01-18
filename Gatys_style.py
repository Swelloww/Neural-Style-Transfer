import torch
import torchvision
from torch import nn
import matplotlib.pyplot as plt
from PIL import Image
import os
import time

# --- 环境设置 ---
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
image_shape = (300, 450)  # (h, w)

# 确保输出目录存在
os.makedirs("output", exist_ok=True)

# --- 图像预处理和后处理 ---
rgb_mean = torch.tensor([0.485, 0.456, 0.406]).to(device)
rgb_std = torch.tensor([0.229, 0.224, 0.225]).to(device)

def preprocess(img_path, image_shape):
    img = Image.open(img_path).convert('RGB')
    transforms = torchvision.transforms.Compose([
        torchvision.transforms.Resize(image_shape),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(mean=rgb_mean.cpu(), std=rgb_std.cpu())])
    return transforms(img).unsqueeze(0).to(device)

def postprocess(img):
    img = img[0]
    # 逆向标准化： (x * std) + mean
    img = torch.clamp(img.permute(1, 2, 0) * rgb_std + rgb_mean, 0, 1)
    return torchvision.transforms.ToPILImage()(img.permute(2, 0, 1))

# --- 加载模型 ---
# 使用 weights 参数代替已废弃的 pretrained
pretrained_net = torchvision.models.vgg19(weights=torchvision.models.VGG19_Weights.IMAGENET1K_V1)
style_layers, content_layers = [0, 5, 10, 19, 28], [25]
net = nn.Sequential(*[pretrained_net.features[i] for i in
                      range(max(content_layers + style_layers) + 1)]).to(device).eval()

# 冻结参数
for param in net.parameters():
    param.requires_grad = False

def extract_features(X, content_layers, style_layers):
    contents = []
    styles = []
    for i in range(len(net)):
        X = net[i](X)
        if i in style_layers:
            styles.append(X)
        if i in content_layers:
            contents.append(X)
    return contents, styles

# --- 损失函数 ---
def content_loss(Y_hat, Y):
    return torch.square(Y_hat - Y.detach()).mean()

def gram(X):
    num_channels, n = X.shape[1], X.numel() // X.shape[1]
    X = X.reshape((num_channels, n))
    return torch.matmul(X, X.T) / (num_channels * n)

def style_loss(Y_hat, gram_Y):
    return torch.square(gram(Y_hat) - gram_Y.detach()).mean()

def tv_loss(Y_hat):
    return 0.5 * (torch.abs(Y_hat[:, :, 1:, :] - Y_hat[:, :, :-1, :]).mean() +
                  torch.abs(Y_hat[:, :, :, 1:] - Y_hat[:, :, :, :-1]).mean())

# --- 训练逻辑 ---
content_weight, style_weight, tv_weight = 1e-2, 5e11, 5

def compute_loss(X, contents_Y_hat, styles_Y_hat, contents_Y, styles_Y_gram):
    contents_l = [content_loss(Y_hat, Y) * content_weight for Y_hat, Y in zip(contents_Y_hat, contents_Y)]
    styles_l = [style_loss(Y_hat, Y) * style_weight for Y_hat, Y in zip(styles_Y_hat, styles_Y_gram)]
    tv_l = tv_loss(X) * tv_weight
    l = sum(styles_l + contents_l + [tv_l])
    return contents_l, styles_l, tv_l, l

def train(content_path, style_path, num_epochs=500, lr=0.15, lr_decay_epoch=100):
    content_X = preprocess(content_path, image_shape)
    style_X = preprocess(style_path, image_shape)
    
    contents_Y, _ = extract_features(content_X, content_layers, style_layers)
    _, styles_Y = extract_features(style_X, content_layers, style_layers)
    styles_Y_gram = [gram(Y) for Y in styles_Y]

    # 初始化合成图像为内容图的副本
    X = nn.Parameter(content_X.clone())
    optimizer = torch.optim.Adam([X], lr=lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, lr_decay_epoch, 0.8)

    print("开始训练...")
    for epoch in range(num_epochs):
        optimizer.zero_grad()
        contents_Y_hat, styles_Y_hat = extract_features(X, content_layers, style_layers)
        contents_l, styles_l, tv_l, l = compute_loss(X, contents_Y_hat, styles_Y_hat, contents_Y, styles_Y_gram)
        
        l.backward()
        optimizer.step()
        scheduler.step()

        if (epoch + 1) % 50 == 0:
            print(f"Epoch [{epoch+1}/{num_epochs}] | Loss: {l.item():.4f} | Content: {sum(contents_l).item():.4f} | Style: {sum(styles_l).item():.4f}")

    return X

# --- 执行主程序 ---
if __name__ == "__main__":
    content_path = './img/megan.png'
    style_path = './img/starry_night.jpg'
    
    if not os.path.exists(content_path) or not os.path.exists(style_path):
        print("错误：请确保照片存在于 './img/' 目录下。")
    else:
        strat_time = time.time()
        output = train(content_path, style_path, num_epochs=1000, lr=0.15)
        final_img = postprocess(output)
        end_time = time.time()
        print(f"总训练时间: {end_time - strat_time:.2f} 秒")
        final_img.save("output/Gatys_stylized_megan_output.png")
        print("训练完成！结果已保存至 output/Gatys_stylized_output.png")