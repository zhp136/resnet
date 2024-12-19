from utils import *
from torch import nn
from datasets import SRDataset
from models import SRResNet,Generator
import time
import torch
import torch.nn.functional as F
import numpy as np

# 模型参数
scaling_factor = 4      # 放大比例
ngpu = 1                # GP数量
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def compute_psnr(img1, img2, max_val=255.0):
    # 计算均方误差 (MSE)
    mse = np.mean((img1 - img2) ** 2)
    if mse == 0:
        return 100  # 完全相同的图像，PSNR达到最大值
    return 10 * np.log10((max_val ** 2) / mse)

def psnr_rgb(img1, img2):
    # 计算RGB图像的PSNR，分别对每个通道计算PSNR
    psnr_values = []
    for i in range(3):  # 对R, G, B三个通道分别计算
        psnr_values.append(compute_psnr(img1[..., i], img2[..., i]))
    return np.mean(psnr_values)  # 可以返回平均值，也可以选择其他合成方法


def compute_ssim(img1, img2, C1=1e-6, C2=1e-6):
    # 计算均值、方差和协方差
    mu1 = np.mean(img1)
    mu2 = np.mean(img2)
    sigma1 = np.var(img1)
    sigma2 = np.var(img2)
    sigma12 = np.cov(img1.flatten(), img2.flatten())[0][1]
    
    # 计算SSIM公式
    numerator = (2 * mu1 * mu2 + C1) * (2 * sigma12 + C2)
    denominator = (mu1 ** 2 + mu2 ** 2 + C1) * (sigma1 + sigma2 + C2)
    return numerator / denominator

def ssim_rgb(img1, img2):
    # 计算RGB图像的SSIM，分别对每个通道计算SSIM
    ssim_values = []
    for i in range(3):  # 对R, G, B三个通道分别计算
        ssim_values.append(compute_ssim(img1[..., i], img2[..., i]))
    return np.mean(ssim_values)  # 返回平均SSIM值


if __name__ == '__main__':
    
    # 测试集目录
    data_folder = "./data/"

    # 加载模型SRResNet 或 SRGAN
    model = torch.load("./results/checkpoint_srresnet.pth")
    model.eval()

    # 定制化数据加载器
    test_dataset = SRDataset(data_folder,
                            split='test',
                            crop_size=0,
                            scaling_factor=4,
                            lr_img_type='imagenet-norm',
                            hr_img_type='[-1, 1]')
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=1,
                                            pin_memory=True)

    # 初始化总和与计数
    total_psnr = 0.0
    total_ssim = 0.0
    num_samples = 0

    # 记录测试时间
    start = time.time()

    with torch.no_grad():
        # 逐批样本进行推理计算
        for i, (lr_imgs, hr_imgs) in enumerate(test_loader):
            
            # 数据移至默认设备
            lr_imgs = lr_imgs.to(device)  # (batch_size (1), 3, w / 4, h / 4), imagenet-normed
            hr_imgs = hr_imgs.to(device)  # (batch_size (1), 3, w, h), in [-1, 1]

            # 前向传播.
            sr_imgs = model(lr_imgs)  # (1, 3, w, h), in [-1, 1]                

            # 计算 PSNR 和 SSIM
            hr_imgs1 = (hr_imgs + 1) * 127.5  # 转换到 [0, 255] 范围
            sr_imgs1 = (sr_imgs + 1) * 127.5  # 转换到 [0, 255] 范围    

            # 转换为numpy格式，方便计算PSNR和SSIM
            hr_imgs1 = hr_imgs1.permute(0, 2, 3, 1).cpu().numpy()
            sr_imgs1 = sr_imgs1.permute(0, 2, 3, 1).cpu().numpy()          

            # 计算PSNR和SSIM
            psnr = psnr_rgb(hr_imgs1[0], sr_imgs1[0])
            ssim = ssim_rgb(hr_imgs1[0], sr_imgs1[0])

            # 累加PSNR和SSIM
            total_psnr += psnr
            total_ssim += ssim
            num_samples += 1

    # 计算并输出平均PSNR和SSIM
    avg_psnr = total_psnr / num_samples
    avg_ssim = total_ssim / num_samples
    print(f'PSNR  {avg_psnr:.3f}')
    print(f'SSIM  {avg_ssim:.3f}')
    print(f'平均单张样本用时  {(time.time()-start)/len(test_dataset):.3f} 秒')

    print("\n")
