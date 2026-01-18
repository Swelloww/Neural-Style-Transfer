import torch
import torchvision.transforms as transforms
from torchvision.models import inception_v3
from scipy import linalg

def get_inception_activations(img_arr, device):
	# 预处理
	transform = transforms.Compose([
		transforms.ToPILImage(),
		transforms.Resize((299, 299)),
		transforms.ToTensor(),
		transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
	])
	img_tensor = transform(img_arr).unsqueeze(0).to(device)
	with torch.no_grad():
		model = inception_v3(pretrained=True, transform_input=False).to(device)
		model.eval()
		# 只取pool3层输出
		features = model.forward_features(img_tensor) if hasattr(model, 'forward_features') else model(img_tensor)
		if isinstance(features, tuple):
			features = features[0]
		if features.ndim == 4:
			features = torch.nn.functional.adaptive_avg_pool2d(features, (1, 1))
			features = features.view(features.size(0), -1)
	return features.cpu().numpy()

def calculate_fid(img1_path, img2_path):
	device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
	img1 = np.array(load_image(img1_path))
	img2 = np.array(load_image(img2_path))
	act1 = get_inception_activations(img1, device)
	act2 = get_inception_activations(img2, device)
	mu1 = act1.mean(axis=0)
	mu2 = act2.mean(axis=0)
	# 单张图片时，np.cov返回标量0，需特殊处理
	def get_cov(act):
		if act.shape[0] == 1:
			# 单张图片，返回单位阵
			return np.eye(act.shape[1])
		else:
			return np.cov(act, rowvar=False)
	sigma1 = get_cov(act1)
	sigma2 = get_cov(act2)
	# 检查NaN/inf
	if not np.all(np.isfinite(mu1)) or not np.all(np.isfinite(mu2)):
		raise ValueError('Mean contains NaN or inf, cannot calculate FID')
	if not np.all(np.isfinite(sigma1)) or not np.all(np.isfinite(sigma2)):
		raise ValueError('Covariance contains NaN or inf, cannot calculate FID')
	# FID公式
	diff = mu1 - mu2
	covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
	if np.iscomplexobj(covmean):
		covmean = covmean.real
	fid = diff.dot(diff) + np.trace(sigma1 + sigma2 - 2 * covmean)
	return float(fid)
# -*- coding: utf-8 -*-
import argparse
import os

def get_image_paths():
	parser = argparse.ArgumentParser(description='Calculate PSNR, SSIM and FID between two images')
	parser.add_argument('--img1', type=str, help='Path to the first image')
	parser.add_argument('--img2', type=str, help='Path to the second image')
	args = parser.parse_args()
	img1 = args.img1
	img2 = args.img2
	if not img1:
		img1 = input('Please enter the path to the first image: ')
	if not img2:
		img2 = input('Please enter the path to the second image: ')
	if not os.path.isfile(img1):
		raise FileNotFoundError(f'Image not found: {img1}')
	if not os.path.isfile(img2):
		raise FileNotFoundError(f'Image not found: {img2}')
	return img1, img2

from PIL import Image
import numpy as np
from skimage.metrics import peak_signal_noise_ratio as compare_psnr
from skimage.metrics import structural_similarity as compare_ssim

def load_image(path):
	img = Image.open(path).convert('RGB')
	return img

def calc_psnr_ssim(img1_path, img2_path):
	img1 = load_image(img1_path)
	img2 = load_image(img2_path)
	# Resize img2 to img1's size
	if img1.size != img2.size:
		print(f"Resizing second image from {img2.size} to {img1.size}")
		img2 = img2.resize(img1.size, Image.BICUBIC)
	arr1 = np.array(img1)
	arr2 = np.array(img2)
	psnr = compare_psnr(arr1, arr2, data_range=255)
	# 兼容skimage不同版本API，自动设置channel_axis或multichannel
	ssim_kwargs = {'data_range': 255}
	import inspect
	sig = inspect.signature(compare_ssim)
	if 'channel_axis' in sig.parameters:
		ssim_kwargs['channel_axis'] = -1
	else:
		ssim_kwargs['multichannel'] = True
	# 自动适配小图像尺寸
	min_side = min(arr1.shape[0], arr1.shape[1])
	if min_side < 7:
		ssim_kwargs['win_size'] = min_side if min_side % 2 == 1 else min_side - 1
	ssim = compare_ssim(arr1, arr2, **ssim_kwargs)
	return psnr, ssim

if __name__ == '__main__':
	# img_output_path = "Boy_Mona_FNN.jpg"
	# img_content_path = "Boy.png"
	# img_style_path = "Mona.jpg"
    img_output_path = "Megan_StarryNight_Gatys.png"
    img_content_path = "Megan.png"
    img_style_path = "Starry_Night.jpg"

    print(f'Output image: {img_output_path}')
    print(f'Content image: {img_content_path}')
    print(f'Style image: {img_style_path}')

    psnr, ssim = calc_psnr_ssim(img_output_path, img_content_path)
    print(f'PSNR: {psnr:.4f} dB')
    print(f'SSIM: {ssim:.4f}')

    fid = calculate_fid(img_output_path, img_style_path)
    print(f'FID: {fid:.4f}')
