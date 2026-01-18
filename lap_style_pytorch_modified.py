#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Masked (foreground/background) style transfer based on a minimal PyTorch port of lap_style.lua

New features added:
- You can specify different style images for background and subject (foreground)
  - -style_image_bg ...  (can be multi-style comma-separated)
  - -style_image_fg ...  (can be multi-style comma-separated)
- The code automatically estimates a subject mask from the content image (default: DeepLabV3)
  - Foreground = the most dominant non-background semantic class
  - Background = 1 - Foreground
- Style loss is computed separately on foreground/background regions using masked Gram matrices

Example:
CUDA_VISIBLE_DEVICES=2 python lap_style_pytorch_modified.py \
  -content_image images/goat.png \
  -style_image_fg images/muse.jpg \
  -style_image_bg images/starry_night.jpg \
  -output_image output/bosong_fg_muse_bg_starry_night.png \
  -content_weight 15 -style_weight 90 -num_iterations 1000 -tv_weight 0.01 \
  -lap_layers 3 -lap_weights 70

Notes:
- If segmentation weights are not cached, torchvision may download them automatically.
- You can force a simpler mask with: -mask_method center
- You can provide your own mask image (white=foreground, black=background): -mask_image path/to/mask.png
"""

import argparse
import os
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import torchvision.models as models
from torchvision.models import VGG19_Weights

# Optional segmentation imports (will fallback if unavailable)
try:
    import torchvision.models.segmentation as seg_models
    from torchvision.models.segmentation import DeepLabV3_ResNet50_Weights
    _HAS_SEG = True
except Exception:
    _HAS_SEG = False

# -----------------------------
# Utilities
# -----------------------------

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def _lanczos_resample():
    # Pillow compatibility across versions
    if hasattr(Image, "Resampling"):
        return Image.Resampling.LANCZOS
    return Image.LANCZOS


def load_image_pil(path, imsize=None):
    img = Image.open(path).convert("RGB")
    if imsize is not None:
        img = img.resize(imsize, _lanczos_resample())
    return img


def pil_to_tensor(pil_img, device):
    t = transforms.ToTensor()(pil_img).unsqueeze(0).to(device)
    return t


def image_loader(path, device, imsize=None):
    pil = load_image_pil(path, imsize=imsize)
    return pil_to_tensor(pil, device)


def normalize_batch(tensor, device):
    mean = torch.tensor(IMAGENET_MEAN, device=device).view(1, 3, 1, 1)
    std = torch.tensor(IMAGENET_STD, device=device).view(1, 3, 1, 1)
    return (tensor - mean) / std


def save_image(tensor, path):
    tensor = tensor.detach().cpu().clamp(0, 1)
    img = transforms.ToPILImage()(tensor.squeeze(0))
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    img.save(path)


def save_mask(mask01, path):
    # mask01: [1,1,H,W] float in [0,1]
    m = mask01.detach().cpu().clamp(0, 1)
    m = (m.squeeze(0).squeeze(0) * 255.0).byte()
    img = Image.fromarray(m.numpy(), mode="L")
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    img.save(path)


def gram_matrix(y):
    (b, ch, h, w) = y.size()
    features = y.view(b, ch, w * h)
    G = torch.bmm(features, features.transpose(1, 2))
    return G / (ch * h * w)


def gram_matrix_masked(feat, mask01):
    """
    Masked Gram matrix.
    feat:  [B, C, H, W]
    mask01:[B, 1, H, W] in [0,1]
    """
    b, c, h, w = feat.shape
    masked = feat * mask01
    features = masked.view(b, c, h * w)
    G = torch.bmm(features, features.transpose(1, 2))
    # normalize by effective pixels
    denom = (mask01.sum(dim=(2, 3), keepdim=True).clamp(min=1.0))  # [B,1,1,1]
    denom = denom.view(b, 1, 1) * c
    return G / denom


def make_laplacian_conv(device):
    lap = torch.zeros(1, 3, 3, 3, device=device)
    lap_kernel = torch.tensor(
        [[0.0, -1.0, 0.0],
         [-1.0, 4.0, -1.0],
         [0.0, -1.0, 0.0]],
        device=device
    )
    for c in range(3):
        lap[0, c] = lap_kernel
    conv = nn.Conv2d(3, 1, kernel_size=3, stride=1, padding=1, bias=False)
    conv.weight.data = lap
    conv.weight.requires_grad = False
    conv.to(device)
    return conv


def total_variation_loss(img):
    return (
        torch.sum(torch.abs(img[:, :, :, :-1] - img[:, :, :, 1:])) +
        torch.sum(torch.abs(img[:, :, :-1, :] - img[:, :, 1:, :]))
    )


def parse_int_list(s):
    if s is None or s == "":
        return []
    return [int(x) for x in s.split(",")]


def parse_float_list(s):
    if s is None or s == "":
        return []
    return [float(x) for x in s.split(",")]


# -----------------------------
# VGG Feature extraction
# -----------------------------

VGG_LAYER_MAP = {
    "relu1_1": 1,
    "relu2_1": 6,
    "relu3_1": 11,
    "relu4_1": 20,
    "relu4_2": 22,
    "relu5_1": 29,
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


# -----------------------------
# Foreground/Background mask
# -----------------------------

def _center_prior_mask(h, w, device):
    # Smooth center prior (values in [0,1])
    yy = torch.linspace(-1.0, 1.0, h, device=device)
    xx = torch.linspace(-1.0, 1.0, w, device=device)
    Y, X = torch.meshgrid(yy, xx, indexing="ij")
    # ellipse-ish gaussian
    r = (X * X) / (0.65 * 0.65) + (Y * Y) / (0.65 * 0.65)
    mask = torch.exp(-3.0 * r).clamp(0.0, 1.0)  # [H,W]
    return mask.view(1, 1, h, w)


def _morph_close_open(mask01, k=9, iters=1):
    # Simple morphological close+open using pooling
    # mask01: [1,1,H,W] float
    for _ in range(iters):
        mask01 = F.max_pool2d(mask01, kernel_size=k, stride=1, padding=k // 2)      # dilate
        mask01 = -F.max_pool2d(-mask01, kernel_size=k, stride=1, padding=k // 2)    # erode
        # open
        mask01 = -F.max_pool2d(-mask01, kernel_size=k, stride=1, padding=k // 2)    # erode
        mask01 = F.max_pool2d(mask01, kernel_size=k, stride=1, padding=k // 2)      # dilate
    return mask01.clamp(0.0, 1.0)


def _blur_mask(mask01, k=7, iters=1):
    # Cheap edge softening
    for _ in range(iters):
        mask01 = F.avg_pool2d(mask01, kernel_size=k, stride=1, padding=k // 2)
    return mask01.clamp(0.0, 1.0)


def load_user_mask(mask_path, device, out_hw):
    # white=foreground, black=background
    pil = Image.open(mask_path).convert("L")
    pil = pil.resize((out_hw[1], out_hw[0]), _lanczos_resample())
    t = transforms.ToTensor()(pil).unsqueeze(0).to(device)  # [1,1,H,W] in [0,1]
    # Make it a bit crisper but still smooth
    t = _blur_mask(t, k=9, iters=1)
    return t.clamp(0.0, 1.0)


def compute_foreground_mask(content_pil, device, out_hw, method="deeplab"):
    """
    Returns mask_fg in [0,1], shape [1,1,H,W].
    Foreground = subject, Background = 1 - Foreground.
    """
    H, W = out_hw

    if method == "none":
        return torch.ones(1, 1, H, W, device=device)  # everything as foreground

    if method == "center":
        mask = _center_prior_mask(H, W, device)
        mask = _morph_close_open(mask, k=9, iters=1)
        mask = _blur_mask(mask, k=9, iters=1)
        return mask

    # default: deeplab
    if not _HAS_SEG:
        # segmentation module missing -> fallback
        mask = _center_prior_mask(H, W, device)
        mask = _blur_mask(mask, k=9, iters=1)
        return mask

    try:
        weights = DeepLabV3_ResNet50_Weights.DEFAULT
        seg_model = seg_models.deeplabv3_resnet50(weights=weights).to(device).eval()

        # Prepare input
        # Prefer official weights transforms if available; otherwise do a minimal safe preprocessing.
        try:
            preproc = weights.transforms()
            inp = preproc(content_pil).unsqueeze(0).to(device)  # [1,3,h,w]
        except Exception:
            # fallback
            inp = transforms.ToTensor()(content_pil)
            inp = transforms.Normalize(mean=weights.meta.get("mean", IMAGENET_MEAN),
                                       std=weights.meta.get("std", IMAGENET_STD))(inp)
            inp = inp.unsqueeze(0).to(device)

        with torch.no_grad():
            out = seg_model(inp)["out"]  # [1,C,h,w]
            labels = out.argmax(1, keepdim=True)  # [1,1,h,w]

        # Choose the dominant non-background class as subject
        lbl = labels.squeeze(0).squeeze(0)  # [h,w]
        num_classes = out.shape[1]
        counts = torch.bincount(lbl.reshape(-1), minlength=num_classes)
        if num_classes <= 1 or counts[1:].max().item() == 0:
            # everything predicted as background
            mask_small = _center_prior_mask(lbl.shape[0], lbl.shape[1], device)
        else:
            best = int(torch.argmax(counts[1:]).item() + 1)
            mask_small = (labels == best).float()  # [1,1,h,w]

        # Resize to content size
        mask = F.interpolate(mask_small, size=(H, W), mode="bilinear", align_corners=False).clamp(0.0, 1.0)

        # refine
        mask = _morph_close_open(mask, k=11, iters=1)
        mask = _blur_mask(mask, k=9, iters=1)

        # sanity fallback
        area = mask.mean().item()
        if area < 0.02 or area > 0.98:
            mask = _center_prior_mask(H, W, device)
            mask = _blur_mask(mask, k=9, iters=1)

        return mask.clamp(0.0, 1.0)

    except Exception:
        # Any failure -> fallback
        mask = _center_prior_mask(H, W, device)
        mask = _blur_mask(mask, k=9, iters=1)
        return mask


# -----------------------------
# Style target builder
# -----------------------------

def _normalize_blend_weights(ws, n):
    if ws and len(ws) == n:
        s = sum(ws)
        if s <= 0:
            return [1.0 / n] * n
        return [w / s for w in ws]
    return [1.0 / n] * n


def build_style_targets(style_imgs, style_blend_weights, cnn, all_indices, style_indices, apply_preprocess):
    """
    style_targets[idx] = blended Gram matrix (unmasked, but compatible with masked Gram normalization)
    """
    if len(style_imgs) == 0:
        return {idx: None for idx in style_indices}

    ws = _normalize_blend_weights(style_blend_weights, len(style_imgs))

    with torch.no_grad():
        feats_list = [get_features(apply_preprocess(s), cnn, all_indices) for s in style_imgs]

    targets = {}
    ones_cache = {}  # cache ones masks by spatial size

    for si in style_indices:
        G_sum = None
        for w, feats in zip(ws, feats_list):
            f = feats[si]
            _, _, h, w_ = f.shape
            key = (h, w_)
            if key not in ones_cache:
                ones_cache[key] = torch.ones(1, 1, h, w_, device=f.device)
            g = gram_matrix_masked(f, ones_cache[key])  # == standard gram scaling
            G_sum = w * g if G_sum is None else (G_sum + w * g)
        targets[si] = G_sum
    return targets


# -----------------------------
# Main
# -----------------------------

def main():
    parser = argparse.ArgumentParser()

    # Content and output
    parser.add_argument("-content_image", required=True)
    parser.add_argument("-output_image", default="out.png")

    # Original (single) style arg kept for backward compatibility:
    parser.add_argument("-style_image", default="")  # if provided, used for both bg/fg unless bg/fg given

    # New: separate styles
    parser.add_argument("-style_image_bg", default="")
    parser.add_argument("-style_image_fg", default="")
    parser.add_argument("-style_blend_weights", default="")       # legacy (applies to both if bg/fg not given)
    parser.add_argument("-style_blend_weights_bg", default="")
    parser.add_argument("-style_blend_weights_fg", default="")

    # Device
    parser.add_argument("-gpu", default="0")

    # Weights
    parser.add_argument("-content_weight", type=float, default=5e0)
    parser.add_argument("-style_weight", type=float, default=1e2)
    parser.add_argument("-style_weight_bg", type=float, default=-1.0)  # if <0 => use style_weight
    parser.add_argument("-style_weight_fg", type=float, default=-1.0)  # if <0 => use style_weight
    parser.add_argument("-tv_weight", type=float, default=1e-2)

    # Laplacian
    parser.add_argument("-lap_weights", default="100")
    parser.add_argument("-lap_layers", default="2")
    parser.add_argument("-lap_nobp", action="store_true")  # kept (not used specially here)

    # Optimization
    parser.add_argument("-num_iterations", type=int, default=1000)
    parser.add_argument("-init", default="random", choices=["random", "content", "image"])
    parser.add_argument("-init_image", default="")
    parser.add_argument("-optimizer", default="lbfgs", choices=["lbfgs", "adam"])
    parser.add_argument("-learning_rate", type=float, default=1.0)
    parser.add_argument("-print_iter", type=int, default=50)
    parser.add_argument("-save_iter", type=int, default=200)

    # Style transfer config
    parser.add_argument("-style_scale", type=float, default=1.0)
    parser.add_argument("-seed", type=int, default=-1)
    parser.add_argument("-content_layers", default="relu4_2")
    parser.add_argument("-style_layers", default="relu1_1,relu2_1,relu3_1,relu4_1,relu5_1")
    parser.add_argument(
        "-preprocess",
        default="caffe",
        choices=["imagenet", "caffe"],
        help="Input preprocessing: imagenet (torchvision) or caffe (BGR*255 - mean)",
    )

    # Mask controls
    parser.add_argument("-mask_method", default="deeplab", choices=["deeplab", "center", "none"])
    parser.add_argument("-mask_image", default="")  # user-provided mask (white=FG)
    parser.add_argument("-mask_output", default="")  # optional path to save computed FG mask

    args = parser.parse_args()

    # Device
    device = torch.device("cpu")
    if args.gpu is not None and args.gpu != "-1":
        try:
            device = torch.device("cuda:" + str(args.gpu.split(",")[0]))
        except Exception:
            device = torch.device("cpu")

    if args.seed is not None and args.seed >= 0:
        torch.manual_seed(args.seed)

    # Load content image (PIL + tensor)
    content_pil = load_image_pil(args.content_image)
    content_img = pil_to_tensor(content_pil, device)

    _, _, H, W = content_img.shape

    # Resolve background/foreground style paths (backward compatible)
    if args.style_image_bg or args.style_image_fg:
        bg_paths = args.style_image_bg.split(",") if args.style_image_bg else []
        fg_paths = args.style_image_fg.split(",") if args.style_image_fg else []
        # If only one provided, reuse it for the other
        if not bg_paths and fg_paths:
            bg_paths = fg_paths
        if not fg_paths and bg_paths:
            fg_paths = bg_paths

        bg_ws = parse_float_list(args.style_blend_weights_bg)
        fg_ws = parse_float_list(args.style_blend_weights_fg)
    else:
        # legacy: single style_image applies to both
        legacy_paths = args.style_image.split(",") if args.style_image else []
        bg_paths = legacy_paths
        fg_paths = legacy_paths
        legacy_ws = parse_float_list(args.style_blend_weights)
        bg_ws = legacy_ws
        fg_ws = legacy_ws

    # Load style images (optionally scaled)
    def load_style_list(paths):
        imgs = []
        for p in paths:
            p = p.strip()
            if not p:
                continue
            pil = load_image_pil(p)
            if args.style_scale != 1.0:
                sw = max(1, int(round(pil.size[0] * args.style_scale)))
                sh = max(1, int(round(pil.size[1] * args.style_scale)))
                pil = pil.resize((sw, sh), _lanczos_resample())
            imgs.append(pil_to_tensor(pil, device))
        return imgs

    style_imgs_bg = load_style_list(bg_paths)
    style_imgs_fg = load_style_list(fg_paths)

    if len(style_imgs_bg) == 0 or len(style_imgs_fg) == 0:
        raise ValueError(
            "You must provide style images. Use -style_image (legacy) "
            "or -style_image_bg and -style_image_fg."
        )

    # VGG19
    cnn_model = models.vgg19(weights=VGG19_Weights.DEFAULT)
    # Try to load converted caffe weights if present (kept from your original script)
    converted_paths = [
        os.path.join("lapstyle", "models", "vgg19_caffe.pth"),
        os.path.join("lapstyle", "models", "VGG_ILSVRC_19_layers.pth"),
        os.path.join("lapstyle", "models", "vgg19_from_caffe.pth"),
    ]
    for p in converted_paths:
        if os.path.exists(p):
            try:
                state = torch.load(p, map_location=device)
                cnn_model.load_state_dict(state)
                print("Loaded converted VGG weights from", p)
                break
            except Exception as e:
                print("Found candidate converted weights at", p, "but failed to load:", e)

    cnn = cnn_model.features.to(device).eval()

    # Layer indices
    content_layer_names = [x.strip() for x in args.content_layers.split(",") if x.strip()]
    style_layer_names = [x.strip() for x in args.style_layers.split(",") if x.strip()]

    content_indices = [VGG_LAYER_MAP.get(n, VGG_LAYER_MAP["relu4_2"]) for n in content_layer_names]
    style_indices = [VGG_LAYER_MAP.get(n, VGG_LAYER_MAP["relu1_1"]) for n in style_layer_names]
    all_indices = sorted(set(content_indices + style_indices))

    # Preprocess function
    def apply_preprocess(tensor):
        if args.preprocess == "imagenet":
            return normalize_batch(tensor, device)
        else:
            # caffe style: RGB -> BGR, *255, subtract mean pixel
            mean_pixel = torch.tensor([103.939, 116.779, 123.68], device=device).view(1, 3, 1, 1)
            t = tensor * 255.0
            t = t[:, [2, 1, 0], :, :]  # BGR
            t = t - mean_pixel
            return t

    # Foreground mask
    if args.mask_image:
        mask_fg = load_user_mask(args.mask_image, device, out_hw=(H, W))
    else:
        mask_fg = compute_foreground_mask(content_pil, device, out_hw=(H, W), method=args.mask_method)

    mask_bg = (1.0 - mask_fg).clamp(0.0, 1.0)

    if args.mask_output:
        save_mask(mask_fg, args.mask_output)
        print("Saved foreground mask to", args.mask_output)

    # Content targets
    with torch.no_grad():
        content_feats = get_features(apply_preprocess(content_img), cnn, all_indices)
    content_targets = {idx: content_feats[idx] for idx in content_indices}

    # Style targets (separately for bg/fg)
    style_targets_bg = build_style_targets(style_imgs_bg, bg_ws, cnn, all_indices, style_indices, apply_preprocess)
    style_targets_fg = build_style_targets(style_imgs_fg, fg_ws, cnn, all_indices, style_indices, apply_preprocess)

    # Laplacian targets
    lap_layers = parse_int_list(args.lap_layers)
    lap_weights = parse_float_list(args.lap_weights)
    if len(lap_weights) == 1 and len(lap_layers) > 1:
        lap_weights = lap_weights * len(lap_layers)

    lap_conv = make_laplacian_conv(device)
    lap_targets = []
    for layer_idx, lw in zip(lap_layers, lap_weights):
        ps = 2 ** layer_idx
        pooled = F.avg_pool2d(content_img, kernel_size=ps, stride=ps)
        lt = lap_conv(pooled)
        lap_targets.append(lt.detach())

    # Init image
    if args.init == "random":
        input_img = torch.randn_like(content_img).mul(0.001).to(device).requires_grad_(True)
    elif args.init == "image":
        if not args.init_image:
            raise ValueError("When -init image, you must provide -init_image.")
        init_img = image_loader(args.init_image, device)
        if init_img.shape != content_img.shape:
            init_img = F.interpolate(init_img, size=(H, W), mode="bilinear", align_corners=False)
        input_img = init_img.clone().requires_grad_(True)
    else:
        input_img = content_img.clone().requires_grad_(True)

    # Optimizer
    if args.optimizer == "lbfgs":
        optimizer = torch.optim.LBFGS([input_img], lr=args.learning_rate)
    else:
        optimizer = torch.optim.Adam([input_img], lr=args.learning_rate)

    mse = nn.MSELoss()

    style_weight_bg = args.style_weight if args.style_weight_bg < 0 else args.style_weight_bg
    style_weight_fg = args.style_weight if args.style_weight_fg < 0 else args.style_weight_fg

    def closure():
        optimizer.zero_grad()

        with torch.no_grad():
            input_img.data.clamp_(0.0, 1.0)

        feats = get_features(apply_preprocess(input_img), cnn, all_indices)

        # Content loss (full image)
        loss_c = 0.0
        for idx in content_indices:
            loss_c = loss_c + args.content_weight * mse(feats[idx], content_targets[idx])

        # Style loss (masked bg/fg)
        loss_s = 0.0
        for idx in style_indices:
            f = feats[idx]
            _, _, h, w = f.shape
            mfg = F.interpolate(mask_fg, size=(h, w), mode="bilinear", align_corners=False).clamp(0.0, 1.0)
            mbg = 1.0 - mfg

            G_fg = gram_matrix_masked(f, mfg)
            G_bg = gram_matrix_masked(f, mbg)

            loss_s = loss_s + style_weight_fg * mse(G_fg, style_targets_fg[idx])
            loss_s = loss_s + style_weight_bg * mse(G_bg, style_targets_bg[idx])

        # Laplacian loss (full image, multi-scale)
        loss_l = 0.0
        for i, (layer_idx, lw) in enumerate(zip(lap_layers, lap_weights)):
            ps = 2 ** layer_idx
            pooled = F.avg_pool2d(input_img, kernel_size=ps, stride=ps)
            lt = lap_conv(pooled)
            loss_l = loss_l + lw * mse(lt, lap_targets[i])

        # TV loss
        loss_tv = args.tv_weight * total_variation_loss(input_img)

        loss = loss_c + loss_s + loss_l + loss_tv
        loss.backward()

        with torch.no_grad():
            input_img.data.clamp_(0.0, 1.0)

        return loss

    print("Running optimization on", device)
    run = [0]

    if args.optimizer == "lbfgs":
        last_loss = [None]

        while run[0] < args.num_iterations:

            def lbfgs_closure():
                run[0] += 1
                loss = closure()
                last_loss[0] = loss

                if run[0] % args.print_iter == 0:
                    print(f"Iteration {run[0]} loss {loss.item():.4f}")

                if run[0] % args.save_iter == 0:
                    save_image(input_img, f"{os.path.splitext(args.output_image)[0]}_{run[0]}.png")

                return loss

            optimizer.step(lbfgs_closure)

    else:
        for it in range(1, args.num_iterations + 1):
            optimizer.step(closure)

            if it % args.print_iter == 0:
                print(f"Iteration {it} done")

            if it % args.save_iter == 0:
                save_image(input_img, f"{os.path.splitext(args.output_image)[0]}_{it}.png")

    save_image(input_img, args.output_image)
    print("Saved output to", args.output_image)


if __name__ == "__main__":
    main()
