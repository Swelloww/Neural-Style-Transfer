import torch
from torchvision import transforms
from models import TransformerNet
from utils import load_image, save_image
import time
import argparse
import os

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def style_transfer(content_filename, style_model_path, output_filename):
    # 拼接路径
    input_dir = "./content_image"
    output_dir = "./outputs"

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Created directory: {output_dir}")

    content_path = os.path.join(input_dir, content_filename)
    output_path = os.path.join(output_dir, output_filename)



    # 加载内容图
    content_image = load_image(content_path, scale=None) # 可以设置 scale 缩放图片
    content_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x.mul(255))
    ])
    content_tensor = content_transform(content_image).unsqueeze(0).to(DEVICE)

    # 加载集成了风格的模型
    print(f"Loading style model: {style_model_path}...")
    with torch.no_grad():
        style_model = TransformerNet()
        state_dict = torch.load(style_model_path)
        style_model.load_state_dict(state_dict)
        style_model.to(DEVICE)
        style_model.eval()

        start_time = time.time()
        # 快速推理
        generated_tensor = style_model(content_tensor)

        end_time = time.time()

    # 保存结果
    save_image(output_path, generated_tensor[0].cpu())
    
    duration = end_time - start_time
    print(f"Done! Saved to {output_path}")
    print(f"Inference time: {duration:.4f} seconds")

if __name__ == "__main__":
    # 配置命令行参数
    parser = argparse.ArgumentParser(description="Fast Style Transfer Inference Script")
    
    parser.add_argument("--content", type=str, required=True, help="Path to the content image")
    parser.add_argument("--model", type=str, required=True, help="Path to the .pth style model")
    parser.add_argument("--output", type=str, default="output.jpg", help="Path to save the result")

    args = parser.parse_args()

    # 将参数传递给函数
    style_transfer(
        content_filename=args.content, 
        style_model_path=args.model, 
        output_filename=args.output
    )
    '''示例： python inference.py --content boy.jpg --model style_swd_integrated_sunrise.pth --output boy_output_oil_painting.jpg'''
    