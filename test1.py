from utils import *
from torch import nn
from models import ResNet,Generator
import time
from PIL import Image

# 测试图像
imgPath = './results/test.jpg'

# 模型参数
scaling_factor = 4      # 放大比例
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


if __name__ == '__main__':

    # 加载模型ResNet
    model = torch.load("./results/checkpoint_resnet.pth")
    model.eval()

    # 加载图像
    img = Image.open(imgPath, mode='r')
    img = img.convert('RGB')

    # 双线性上采样
    Bicubic_img = img.resize((int(img.width * scaling_factor),int(img.height * scaling_factor)),Image.BICUBIC)
    Bicubic_img.save('./results/test_bicubic.jpg')

    # 图像预处理
    lr_img = convert_image(img, source='pil', target='imagenet-norm')
    lr_img.unsqueeze_(0)

    # 记录时间
    start = time.time()

    # 转移数据至设备
    lr_img = lr_img.to(device)  # (1, 3, w, h ), imagenet-normed

    # 模型推理
    with torch.no_grad():
        sr_img = model(lr_img).squeeze(0).cpu().detach()  # (1, 3, w*scale, h*scale), in [-1, 1]   
        sr_img = convert_image(sr_img, source='[-1, 1]', target='pil')
        sr_img.save('./results/test_resnet.jpg')

    print('用时  {:.3f} 秒'.format(time.time()-start))

