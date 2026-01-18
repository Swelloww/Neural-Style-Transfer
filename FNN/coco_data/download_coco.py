import os
import requests

# 配置
SAVE_DIR = "train2014"  # 您的目标文件夹
NUM_IMAGES = 2000        # 您需要的张数（如 2000 张）

if not os.path.exists(SAVE_DIR):
    os.makedirs(SAVE_DIR)

print(f"开始下载 {NUM_IMAGES} 张 COCO 图片到 {SAVE_DIR}...")

# COCO 2014 的图片 ID 规律：通常是 12 位数字
# 我们从 1 开始尝试下载
downloaded = 0
current_id = 1

while downloaded < NUM_IMAGES:
    # 构造 COCO 2014 的官方图片 URL
    img_name = f"COCO_train2014_{current_id:012d}.jpg"
    url = f"http://images.cocodataset.org/train2014/{img_name}"
    
    try:
        # 使用 stream=True 节省内存
        response = requests.get(url, timeout=5, stream=True)
        if response.status_code == 200:
            with open(os.path.join(SAVE_DIR, img_name), 'wb') as f:
                for chunk in response.iter_content(1024):
                    f.write(chunk)
            downloaded += 1
            if downloaded % 100 == 0:
                print(f"进度: {downloaded}/{NUM_IMAGES}")
    except:
        pass # 忽略不存在的 ID 或网络波动
    
    current_id += 1

print("下载完成！")