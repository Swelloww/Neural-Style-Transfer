import argparse
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from torchvision import transforms
from PIL import Image

# --- 新增：正则化模块 ---

class Regularizer:
    @staticmethod
    def get_tv_loss(img):
        """Total Variation Loss: 抑制高频噪声"""
        w_variance = torch.sum(torch.pow(img[:, :, :, :-1] - img[:, :, :, 1:], 2))
        h_variance = torch.sum(torch.pow(img[:, :, :-1, :] - img[:, :, 1:, :], 2))
        return w_variance + h_variance

    @staticmethod
    def get_laplacian_kernel(device):
        """创建拉普拉斯卷积核"""
        kernel = torch.tensor([[0, 1, 0], [1, -4, 1], [0, 1, 0]], dtype=torch.float32)
        kernel = kernel.view(1, 1, 3, 3).repeat(3, 1, 1, 1)
        return kernel.to(device)

    @classmethod
    def get_laplace_loss(cls, input_img, content_img, device):
        """Laplacian Loss: 保持图像的二阶梯度一致，防止平滑区产生纹理"""
        kernel = cls.get_laplacian_kernel(device)
        # 对输入图和内容图分别做拉普拉斯变换
        input_lap = F.conv2d(input_img, kernel, groups=3, padding=1)
        content_lap = F.conv2d(content_img, kernel, groups=3, padding=1)
        return F.mse_loss(input_lap, content_lap)

# --- 原始代码逻辑保持不变，部分封装调整 ---

class ImageProcessor:
    def __init__(self, device, mode='caffe'):
        self.device = device
        self.mode = mode
        self.imagenet_mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
        self.imagenet_std = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)
        self.caffe_mean = torch.tensor([103.939, 116.779, 123.68], device=device).view(1, 3, 1, 1)

    def load(self, path):
        img = Image.open(path).convert('RGB')
        tensor = transforms.ToTensor()(img).unsqueeze(0).to(self.device)
        return tensor

    def preprocess(self, tensor):
        if self.mode == 'imagenet':
            return (tensor - self.imagenet_mean) / self.imagenet_std
        else:
            t = tensor[:, [2, 1, 0], :, :] * 255.0
            return t - self.caffe_mean

    def postprocess_and_save(self, tensor, path):
        tensor = tensor.detach().cpu().clamp(0, 1).squeeze(0)
        img = transforms.ToPILImage()(tensor)
        os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
        img.save(path)

class PrecisionW2Engine:
    def __init__(self, device):
        self.device = device

    def get_w2_loss(self, feat_i, feat_s):
        C = feat_i.size(1)
        fi = feat_i.view(C, -1)
        fs = feat_s.view(C, -1).detach()
        mu_i = fi.mean(1)
        mu_s = fs.mean(1)
        loss_mean = torch.norm(mu_i - mu_s, p=2)
        si = torch.mm(fi - mu_i.unsqueeze(1), (fi - mu_i.unsqueeze(1)).t()) / (fi.size(1)-1)
        ss = torch.mm(fs - mu_s.unsqueeze(1), (fs - mu_s.unsqueeze(1)).t()) / (fs.size(1)-1)
        loss_cov = torch.norm(si - ss, p='fro')
        return loss_mean + loss_cov

class FeatureExtractor(nn.Module):
    VGG_MAP = {'relu1_1': 1, 'relu2_1': 6, 'relu3_1': 11, 'relu4_1': 20, 'relu4_2': 22, 'relu5_1': 29}
    def __init__(self, device):
        super().__init__()
        vgg = models.vgg19(pretrained=True).features
        self.model = vgg.to(device).eval()
        for param in self.model.parameters():
            param.requires_grad = False

    def forward(self, x, layer_names):
        features = {}
        indices = {self.VGG_MAP[name]: name for name in layer_names}
        max_idx = max(indices.keys())
        for i, layer in enumerate(self.model):
            x = layer(x)
            if i in indices:
                features[indices[i]] = x
            if i >= max_idx: break
        return features

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-style_image', required=True)
    parser.add_argument('-content_image', required=True)
    parser.add_argument('-iter', type=int, default=500)
    parser.add_argument('-sw', type=float, default=5e4, help='Style Weight')
    parser.add_argument('-cw', type=float, default=1.0, help='Content Weight')
    parser.add_argument('-tvw', type=float, default=1e-3, help='TV Loss Weight')
    parser.add_argument('-lw', type=float, default=1e1, help='Laplace Loss Weight')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    proc = ImageProcessor(device)
    ext = FeatureExtractor(device)
    engine = PrecisionW2Engine(device)
    reg = Regularizer()

    c_img = proc.load(args.content_image)
    s_img = proc.load(args.style_image)

    style_layers = ['relu1_1', 'relu2_1', 'relu3_1', 'relu4_1', 'relu5_1']
    content_layers = ['relu4_2']

    with torch.no_grad():
        c_prep = proc.preprocess(c_img)
        s_prep = proc.preprocess(s_img)
        c_targets = ext(c_prep, content_layers)
        s_targets = ext(s_prep, style_layers)

    input_img = c_img.clone().requires_grad_(True)
    optimizer = torch.optim.LBFGS([input_img], lr=1, max_iter=args.iter)

    run = [0]
    def closure():
        optimizer.zero_grad()
        input_prep = proc.preprocess(input_img)
        feats = ext(input_prep, style_layers + content_layers)
        
        # 1. Content Loss
        loss_c = 0
        for n in content_layers:
            loss_c += F.mse_loss(feats[n], c_targets[n])
        
        # 2. Style Loss (WD)
        loss_s = 0
        for n in style_layers:
            loss_s += engine.get_w2_loss(feats[n], s_targets[n])
        
        # 3. TV Loss (平滑空间噪声)
        loss_tv = reg.get_tv_loss(input_img)
        
        # 4. Laplace Loss (保持原图平滑结构的梯度)
        loss_lap = reg.get_laplace_loss(input_img, c_img, device)
            
        total_loss = (loss_c * args.cw + 
                      loss_s * args.sw + 
                      loss_tv * args.tvw + 
                      loss_lap * args.lw)
        
        total_loss.backward()
        
        run[0] += 1
        if run[0] % 50 == 0:
            print(f"Step {run[0]} | L_total: {total_loss.item():.2f} | L_tv: {loss_tv.item():.2f} | L_lap: {loss_lap.item():.2f}")
        return total_loss

    optimizer.step(closure)
    
    input_img.data.clamp_(0, 1)
    proc.postprocess_and_save(input_img, "output/output_refined.png")
    print("Optimization finished. Image saved to output/output_refined.png")

if __name__ == '__main__':
    main()