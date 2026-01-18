# Neural-Style-Transfer 项目简介

本项目实现了五种主流的神经风格迁移方法，分别为：
- Gatys-style（经典神经风格迁移）
- Lapstyle（拉普拉斯正则化风格迁移）
- CustomSection（分区/掩膜风格迁移，支持前景/背景分离）
- WD-lapstyle（Wasserstein距离+拉普拉斯正则化）
- FNN（快速前馈神经网络风格迁移，支持Sliced Wasserstein Loss）

## 依赖环境

请先安装 requirements.txt 中的依赖：
```
pip install -r requirements.txt
```

---

## 1. Gatys-style（Gatys神经风格迁移）

**原理简介**  
基于 VGG19 网络提取内容和风格特征，通过优化一张初始图片，使其内容特征接近内容图、风格特征（Gram矩阵）接近风格图。损失函数包括内容损失、风格损失和全变分（TV）损失。

**使用方法**  
修改 `Gatys_style.py` 中的图片路径，直接运行：
```
python Gatys_style.py
```
输出结果保存在 `output/Gatys_stylized_output.png`。

---

## 2. Lapstyle（拉普拉斯正则化风格迁移）

**原理简介**  
在经典风格迁移基础上，增加拉普拉斯正则项，约束生成图像的二阶梯度，抑制过度平滑区域产生纹理，提升结构保真度。

**使用方法**  
命令行参数灵活，示例：
```
python lap_style.py -content_image images/megan.png -style_image images/starry_night.jpg -output_image output/lap_stylized.png
```
可调参数包括内容/风格/拉普拉斯/TV权重、迭代次数、初始化方式等。

---

## 3. CustomSection（分区/掩膜风格迁移）

**原理简介**  
支持对前景和背景分别指定不同风格图，自动或手动分割前景掩膜，分别计算前景/背景的风格损失，提升局部风格迁移的灵活性。支持 DeepLabV3 自动分割、中心先验或自定义掩膜。

**使用方法**  
示例命令：
```
python CustomSection_lap_style.py \
	-content_image images/goat.png \
	-style_image_fg images/muse.jpg \
	-style_image_bg images/starry_night.jpg \
	-output_image output/fg_muse_bg_starry_night.png
```
可选参数详见文件头注释，支持多风格融合、掩膜保存、分区权重等。

---

## 4. WD-lapstyle（Wasserstein距离+拉普拉斯正则化）

**原理简介**  
将风格损失替换为特征分布的Wasserstein距离（均值+协方差），更好地对齐风格分布，同时结合拉普拉斯和TV正则，提升风格迁移质量。

**使用方法**  
示例命令：
```
python WD_lap_style.py -content_image images/megan.png -style_image images/starry_night.jpg
```
可调参数包括内容/风格/拉普拉斯/TV权重、迭代次数等。

---

## 5. FNN（快速前馈神经网络风格迁移）

**原理简介**  
训练一个前馈网络（TransformerNet），一次前向推理即可完成风格迁移。训练时采用内容损失、Sliced Wasserstein风格损失和TV损失。推理速度极快，适合批量处理。

**训练方法**  
在 `FNN/` 目录下运行：
```
python train.py
```
训练数据需放在 `FNN/coco_data/` 下，风格图路径和参数可在 `train.py` 中修改。

**推理方法**  
```
python inference.py --content boy.jpg --model style_swd_integrated_sunrise.pth --output test_output_oil_painting.jpg
```
模型文件可用训练得到或使用已集成的权重。

---

如需详细参数说明和自定义用法，请参考各 .py 文件头部注释和 argparse 参数定义。