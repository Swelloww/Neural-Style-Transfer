#!/usr/bin/env python3
"""
Minimal PyTorch port of lap_style.lua

Features:
- loads VGG19 from torchvision
- computes content, style, TV and laplacian losses
- supports multi-style blending, lap_layers and lap_weights

This is a compact, pragmatic port to help migrate from Torch7 to PyTorch.

usage example:
python lap_style_pytorch.py -content_image images/megan.png -style_image images/starry_night.jpg -output_image output/megan_starry_night.png -content_weight 40 -lap_layers 5 -lap_weights 100 -num_iterations 500

"""
import argparse
import os
from PIL import Image
import math
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import torchvision.models as models
import torchvision
import numpy as np


def image_loader(path, device, imsize=None):
    img = Image.open(path).convert('RGB')
    if imsize is not None:
        img = img.resize(imsize, Image.LANCZOS)
    loader = transforms.Compose([
        transforms.ToTensor(),
    ])
    tensor = loader(img).unsqueeze(0).to(device)
    return tensor


IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def normalize_batch(tensor, device):
    """Normalize a batch of images with ImageNet statistics."""
    mean = torch.tensor(IMAGENET_MEAN, device=device).view(1, 3, 1, 1)
    std = torch.tensor(IMAGENET_STD, device=device).view(1, 3, 1, 1)
    return (tensor - mean) / std


def save_image(tensor, path):
    tensor = tensor.detach().cpu().clamp(0, 1)
    img = transforms.ToPILImage()(tensor.squeeze(0))
    os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
    img.save(path)


def gram_matrix(y):
    (b, ch, h, w) = y.size()
    features = y.view(b, ch, w * h)
    G = torch.bmm(features, features.transpose(1, 2))
    return G / (ch * h * w)


VGG_LAYER_MAP = {
    'relu1_1': 1,
    'relu2_1': 6,
    'relu3_1': 11,
    'relu4_1': 20,
    'relu4_2': 22,
    'relu5_1': 29,
}


def get_features(x, cnn, layer_indices):
    features = {}
    current = x
    max_idx = max(layer_indices) if layer_indices else -1
    for i, layer in enumerate(cnn):
        current = layer(current)
        if i in layer_indices:
            features[i] = current
        if i >= max_idx:
            break
    return features


def make_laplacian_conv(device):
    # 3x3 laplacian kernel applied per-channel -> reduce to 1 channel
    lap = torch.zeros(1, 3, 3, 3, device=device)
    lap_kernel = torch.tensor([[0.0, -1.0, 0.0], [-1.0, 4.0, -1.0], [0.0, -1.0, 0.0]], device=device)
    for c in range(3):
        lap[0, c] = lap_kernel
    conv = nn.Conv2d(3, 1, kernel_size=3, stride=1, padding=1, bias=False)
    conv.weight.data = lap
    conv.weight.requires_grad = False
    conv.to(device)
    return conv


def total_variation_loss(img):
    return torch.sum(torch.abs(img[:, :, :, :-1] - img[:, :, :, 1:])) + torch.sum(torch.abs(img[:, :, :-1, :] - img[:, :, 1:, :]))


def parse_tuple_list(s):
    if s is None or s == '':
        return []
    return [int(x) for x in s.split(',')]


def main():
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

    device = torch.device('cpu')
    if args.gpu is not None and args.gpu != '-1':
        try:
            device = torch.device('cuda:' + str(args.gpu.split(',')[0]))
        except Exception:
            device = torch.device('cpu')

    if args.seed and args.seed >= 0:
        torch.manual_seed(args.seed)

    # load images
    content_img = image_loader(args.content_image, device)
    style_paths = args.style_image.split(',') if args.style_image else []
    style_imgs = [image_loader(p, device) for p in style_paths]

    # use VGG19
    cnn_model = models.vgg19(pretrained=True)
    # If a converted PyTorch VGG (from Caffe) exists under lapstyle/models, try to load it
    converted_paths = [
        os.path.join('lapstyle','models','vgg19_caffe.pth'),
        os.path.join('lapstyle','models','VGG_ILSVRC_19_layers.pth'),
        os.path.join('lapstyle','models','vgg19_from_caffe.pth'),
    ]
    converted_loaded = False
    for p in converted_paths:
        if os.path.exists(p):
            try:
                state = torch.load(p, map_location=device)
                cnn_model.load_state_dict(state)
                converted_loaded = True
                print('Loaded converted VGG weights from', p)
                break
            except Exception as e:
                print('Found candidate converted weights at', p, 'but failed to load:', e)
    cnn = cnn_model.features.to(device).eval()

    # prepare layer index mapping
    content_layer_names = args.content_layers.split(',')
    style_layer_names = args.style_layers.split(',')
    content_indices = [VGG_LAYER_MAP.get(n, VGG_LAYER_MAP['relu4_2']) for n in content_layer_names]
    style_indices = [VGG_LAYER_MAP.get(n, VGG_LAYER_MAP['relu1_1']) for n in style_layer_names]
    all_indices = sorted(set(content_indices + style_indices))

    # compute targets
    # choose preprocessing function
    def apply_preprocess(tensor):
        if args.preprocess == 'imagenet':
            return normalize_batch(tensor, device)
        else:
            # caffe style: RGB->[B,G,R], *255, subtract mean pixel
            # tensor expected in [0,1]
            mean_pixel = torch.tensor([103.939, 116.779, 123.68], device=device).view(1,3,1,1)
            # convert to 0..255 and BGR
            t = tensor * 255.0
            t = t[:, [2,1,0], :, :]
            t = t * 1.0
            t = t - mean_pixel
            return t

    with torch.no_grad():
        content_feats = get_features(apply_preprocess(content_img), cnn, all_indices)
        style_feats_list = [get_features(apply_preprocess(s), cnn, all_indices) for s in style_imgs]

    content_targets = {idx: content_feats[idx] for idx in content_indices}
    style_targets = {}
    # blend style gram matrices
    style_blend_weights = [float(x) for x in args.style_blend_weights.split(',')] if args.style_blend_weights else []
    if style_blend_weights and len(style_blend_weights) != len(style_imgs):
        style_blend_weights = []
    if not style_blend_weights:
        style_blend_weights = [1.0] * len(style_imgs)
    ssum = sum(style_blend_weights)
    style_blend_weights = [w / ssum for w in style_blend_weights]

    for si in style_indices:
        # sum weighted gram matrices
        G = None
        for w, s_feats in zip(style_blend_weights, style_feats_list):
            g = gram_matrix(s_feats[si])
            if G is None:
                G = w * g
            else:
                G = G + w * g
        style_targets[si] = G

    # laplacian targets
    lap_layers = parse_tuple_list(args.lap_layers)
    lap_weights = [float(x) for x in args.lap_weights.split(',')]
    if len(lap_weights) == 1 and len(lap_layers) > 1:
        lap_weights = lap_weights * len(lap_layers)

    lap_conv = make_laplacian_conv(device)
    lap_targets = []
    for layer_idx, lw in zip(lap_layers, lap_weights):
        ps = 2 ** layer_idx
        pooled = F.avg_pool2d(content_img, kernel_size=ps, stride=ps)
        lt = lap_conv(pooled)
        lap_targets.append(lt.detach())

    # initialize input image
    if args.init == 'random':
        input_img = torch.randn_like(content_img).mul(0.001).to(device).requires_grad_(True)
    else:
        if args.init_image:
            init_img = image_loader(args.init_image, device)
            input_img = init_img.clone().requires_grad_(True)
        else:
            input_img = content_img.clone().requires_grad_(True)

    # optimizer
    if args.optimizer == 'lbfgs':
        optimizer = torch.optim.LBFGS([input_img], lr=args.learning_rate)
    else:
        optimizer = torch.optim.Adam([input_img], lr=args.learning_rate)

    mse = nn.MSELoss()

    def closure():
        optimizer.zero_grad()
        # ensure input stays in a reasonable numeric range before forward
        with torch.no_grad():
            input_img.data.clamp_(0.0, 1.0)
        feats = get_features(apply_preprocess(input_img), cnn, all_indices)
        loss_c = 0.0
        for idx in content_indices:
            loss_c = loss_c + args.content_weight * mse(feats[idx], content_targets[idx])

        loss_s = 0.0
        for idx in style_indices:
            G = gram_matrix(feats[idx])
            loss_s = loss_s + args.style_weight * mse(G, style_targets[idx])

        loss_l = 0.0
        for i, (layer_idx, lw) in enumerate(zip(lap_layers, lap_weights)):
            ps = 2 ** layer_idx
            pooled = F.avg_pool2d(input_img, kernel_size=ps, stride=ps)
            lt = lap_conv(pooled)
            loss_l = loss_l + lw * mse(lt, lap_targets[i])

        loss_tv = args.tv_weight * total_variation_loss(input_img)

        loss = loss_c + loss_s + loss_l + loss_tv
        loss.backward()
        # keep the image in a reasonable range to avoid NaNs/exploding values
        with torch.no_grad():
            input_img.data.clamp_(0.0, 1.0)
        return loss

    print('Running optimization on', device)
    run = [0]
    if args.optimizer == 'lbfgs':
        # LBFGS may call the closure multiple times per step; increment
        # the iteration counter on each closure invocation to keep
        # behavior similar to previous implementation (avoids extra
        # optimizer.step calls which slow down runs).
        last_loss = [None]
        while run[0] < args.num_iterations:
            def lbfgs_closure():
                run[0] += 1
                loss = closure()
                last_loss[0] = loss
                if run[0] % args.print_iter == 0:
                    print(f'Iteration {run[0]} loss {loss.item():.4f}')
                if run[0] % args.save_iter == 0:
                    save_image(input_img, f"{os.path.splitext(args.output_image)[0]}_{run[0]}.png")
                return loss
            optimizer.step(lbfgs_closure)
    else:
        for it in range(1, args.num_iterations + 1):
            optimizer.step(closure)
            if it % args.print_iter == 0:
                print(f'Iteration {it} done')
            if it % args.save_iter == 0:
                save_image(input_img, f"{os.path.splitext(args.output_image)[0]}_{it}.png")

    # final save
    save_image(input_img, args.output_image)
    print('Saved output to', args.output_image)


if __name__ == '__main__':
    main()
