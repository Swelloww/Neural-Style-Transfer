import argparse
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from torchvision import transforms
from PIL import Image

# --- 1. 图像处理模块 ---

class ImageProcessor:
    def __init__(self, device, mode='caffe'):
        self.device = device
        self.mode = mode
        self.imagenet_mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
        self.imagenet_std = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)
        self.caffe_mean = torch.tensor([103.939, 116.779, 123.68], device=device).view(1, 3, 1, 1)

    def load(self, path, imsize=None):
        img = Image.open(path).convert('RGB')
        if imsize:
            img = img.resize(imsize, Image.LANCZOS)
        tensor = transforms.ToTensor()(img).unsqueeze(0).to(self.device)
        return tensor

    def preprocess(self, tensor):
        if self.mode == 'imagenet':
            return (tensor - self.imagenet_mean) / self.imagenet_std
        else:
            # Caffe: RGB -> BGR, [0,1]->[0,255], subtract mean
            t = tensor[:, [2, 1, 0], :, :] * 255.0
            return t - self.caffe_mean

    def postprocess_and_save(self, tensor, path):
        tensor = tensor.detach().cpu().clamp(0, 1).squeeze(0)
        img = transforms.ToPILImage()(tensor)
        os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
        img.save(path)

# --- 2. 特征提取模块 ---

class FeatureExtractor(nn.Module):
    VGG_MAP = {'relu1_1': 1, 'relu2_1': 6, 'relu3_1': 11, 'relu4_1': 20, 'relu4_2': 22, 'relu5_1': 29}

    def __init__(self, device, weights_path=None):
        super().__init__()
        vgg = models.vgg19(pretrained=True).features
        if weights_path and os.path.exists(weights_path):
            vgg.load_state_dict(torch.load(weights_path, map_location=device))
        
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
            if i >= max_idx:
                break
        return features

# --- 3. 损失函数与模型核心 ---

class StyleTransferEngine:
    def __init__(self, extractor, processor, config):
        self.extractor = extractor
        self.processor = processor
        self.cfg = config
        self.mse = nn.MSELoss()
        self.lap_conv = self._make_laplacian_conv()

    def _make_laplacian_conv(self):
        lap_kernel = torch.tensor([[0., -1., 0.], [-1., 4., -1.], [0., -1., 0.]], device=self.processor.device)
        conv = nn.Conv2d(3, 1, 3, padding=1, bias=False)
        conv.weight.data = lap_kernel.expand(1, 3, 3, 3)
        conv.weight.requires_grad = False
        return conv.to(self.processor.device)

    @staticmethod
    def gram_matrix(y):
        (b, ch, h, w) = y.size()
        features = y.view(b, ch, w * h)
        return torch.bmm(features, features.transpose(1, 2)) / (ch * h * w)

    def get_content_loss(self, input_feats, targets):
        loss = 0
        for name in self.cfg.content_layers:
            loss += self.mse(input_feats[name], targets[name])
        return loss * self.cfg.content_weight

    def get_style_loss(self, input_feats, targets):
        loss = 0
        for name in self.cfg.style_layers:
            loss += self.mse(self.gram_matrix(input_feats[name]), targets[name])
        return loss * self.cfg.style_weight

    def get_laplacian_loss(self, input_img, targets):
        loss = 0
        for (layer_idx, weight), target in zip(self.cfg.lap_params, targets):
            ps = 2 ** layer_idx
            pooled = F.avg_pool2d(input_img, kernel_size=ps, stride=ps)
            loss += weight * self.mse(self.lap_conv(pooled), target)
        return loss

    def get_tv_loss(self, img):
        return self.cfg.tv_weight * (torch.sum(torch.abs(img[:, :, :, :-1] - img[:, :, :, 1:])) + 
                                     torch.sum(torch.abs(img[:, :, :-1, :] - img[:, :, 1:, :])))

# --- 4. 主程序逻辑 ---

def main():
    # ... (此处省略 argparse 解析过程，假设已解析为 args) ...
    parser = argparse.ArgumentParser()
    parser.add_argument('-style_image')
    parser.add_argument('-style_blend_weights', default='')
    parser.add_argument('-content_image')
    parser.add_argument('-gpu', default='0')
    parser.add_argument('-content_weight', type=float, default=5e0)
    parser.add_argument('-style_weight', type=float, default=1e2)
    parser.add_argument('-lap_weights', default='100')
    parser.add_argument('-lap_layers', default='2')
    parser.add_argument('-lap_nobp', action='store_true')
    parser.add_argument('-tv_weight', type=float, default=1e-3)
    parser.add_argument('-num_iterations', type=int, default=1000)
    parser.add_argument('-init', default='random')
    parser.add_argument('-init_image', default='')
    parser.add_argument('-optimizer', default='lbfgs')
    parser.add_argument('-learning_rate', type=float, default=1.0)
    parser.add_argument('-print_iter', type=int, default=50)
    parser.add_argument('-save_iter', type=int, default=100)
    parser.add_argument('-output_image', default='out.png')
    parser.add_argument('-style_scale', type=float, default=1.0)
    parser.add_argument('-seed', type=int, default=-1)
    parser.add_argument('-content_layers', default='relu4_2')
    parser.add_argument('-style_layers', default='relu1_1,relu2_1,relu3_1,relu4_1,relu5_1')
    parser.add_argument('-preprocess', default='caffe', choices=['imagenet','caffe'],
                        help='Input preprocessing: imagenet (torchvision) or caffe (BGR*255 - mean)')
    args = parser.parse_args()
    # 辅助解析：将 lap_layers 和 lap_weights 合并为配置
    args.content_layers = args.content_layers.split(',')
    args.style_layers = args.style_layers.split(',')
    lap_layers = [int(x) for x in args.lap_layers.split(',')]
    lap_weights = [float(x) for x in args.lap_weights.split(',')]
    if len(lap_weights) == 1: lap_weights *= len(lap_layers)
    args.lap_params = list(zip(lap_layers, lap_weights))

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    processor = ImageProcessor(device, mode=args.preprocess)
    extractor = FeatureExtractor(device)
    engine = StyleTransferEngine(extractor, processor, args)

    # 1. 准备目标特征
    content_img = processor.load(args.content_image)
    with torch.no_grad():
        c_prep = processor.preprocess(content_img)
        content_targets = extractor(c_prep, args.content_layers)
        
        # Laplacian targets
        lap_targets = []
        for layer_idx, _ in args.lap_params:
            ps = 2 ** layer_idx
            pooled = F.avg_pool2d(content_img, kernel_size=ps, stride=ps)
            lap_targets.append(engine.lap_conv(pooled).detach())

        # Style targets (此处简化了多风格融合逻辑，可根据需要保留原有的 blend 逻辑)
        style_img = processor.load(args.style_image)
        s_prep = processor.preprocess(style_img)
        s_feats = extractor(s_prep, args.style_layers)
        style_targets = {name: engine.gram_matrix(feat) for name, feat in s_feats.items()}

    # 2. 初始化图
    input_img = content_img.clone().requires_grad_(True)
    optimizer = torch.optim.LBFGS([input_img], lr=args.learning_rate)

    # 3. 优化循环
    run = [0]
    while run[0] <= args.num_iterations:
        def closure():
            optimizer.zero_grad()
            with torch.no_grad():
                input_img.data.clamp_(0, 1)
            
            input_feats = extractor(processor.preprocess(input_img), args.content_layers + args.style_layers)
            
            l_c = engine.get_content_loss(input_feats, content_targets)
            l_s = engine.get_style_loss(input_feats, style_targets)
            l_l = engine.get_laplacian_loss(input_img, lap_targets)
            l_tv = engine.get_tv_loss(input_img)
            
            total_loss = l_c + l_s + l_l + l_tv
            total_loss.backward()
            
            run[0] += 1
            if run[0] % args.print_iter == 0:
                print(f"Iter {run[0]}: {total_loss.item():.4f}")
            return total_loss

        optimizer.step(closure)

    processor.postprocess_and_save(input_img, args.output_image)

if __name__ == '__main__':
    main()