import argparse
import torch.backends.cudnn as cudnn
import torch
from torch import nn
from torchvision.utils import make_grid
from torch.utils.data import random_split
from torch.utils.tensorboard import SummaryWriter
from models import SRResNet
from datasets import SRDataset
from utils import *


# 数据集参数
data_folder = './data/'          # 数据存放路径
crop_size = 96      # 高分辨率图像裁剪尺寸
scaling_factor = 4  # 放大比例

# 模型参数
large_kernel_size = 9   # 第一层卷积和最后一层卷积的核大小
small_kernel_size = 3   # 中间层卷积的核大小
n_channels = 64         # 中间层通道数
n_blocks = 16           # 残差模块数量

# 设备参数
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
ngpu = 1           # 用来运行的gpu数量

cudnn.benchmark = True # 对卷积进行加速

def parse_args():
    parser = argparse.ArgumentParser(description="lr-hr")
    parser.add_argument('--batch_size', type=int, default=400, help='Batch size for training')
    parser.add_argument('--epochs', type=int, default=130, help='Number of epochs to train')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
    return parser.parse_args()

def main():

    conf = parse_args()

    # 初始化
    model = SRResNet(large_kernel_size=large_kernel_size,
                        small_kernel_size=small_kernel_size,
                        n_channels=n_channels,
                        n_blocks=n_blocks,
                        scaling_factor=scaling_factor)
    # 初始化优化器
    optimizer = torch.optim.Adam(params=filter(lambda p: p.requires_grad, model.parameters()),lr=conf.lr)

    # 迁移至默认设备进行训练
    model = model.to(device)
    criterion = nn.MSELoss().to(device)
    
    if torch.cuda.is_available() and ngpu > 1:
        model = nn.DataParallel(model, device_ids=list(range(ngpu)))

    # 定制化的dataloader
    train_dataset = SRDataset(data_folder,split='train',
                              crop_size=crop_size,
                              scaling_factor=scaling_factor,
                              lr_img_type='imagenet-norm',
                              hr_img_type='[-1, 1]')

    # 计算训练集和验证集的大小
    dataset_size = len(train_dataset)
    train_size = int(0.7 * dataset_size)  # 70%用于训练
    val_size = dataset_size - train_size  # 剩余的30%用于验证

    # 使用random_split划分数据集
    train_dataset, val_dataset = random_split(train_dataset, [train_size, val_size])

    # 创建训练集和验证集的DataLoader
    train_loader = torch.utils.data.DataLoader(train_dataset,
                                            batch_size=conf.batch_size,
                                            shuffle=True,
                                            pin_memory=True)

    val_loader = torch.utils.data.DataLoader(val_dataset,
                                            batch_size=conf.batch_size,
                                            shuffle=True, 
                                            pin_memory=True)

    modelweightfile = f'results/checkpoint_srresnet.pth'

    def train(model, loss, train_dataloader, optimizer, epoch):
        model.train()
        all_loss = 0
        for i, (lr_imgs, hr_imgs) in enumerate(train_dataloader):
            # 数据移至默认设备进行训练
            lr_imgs = lr_imgs.to(device)  # (batch_size (N), 3, 24, 24), imagenet-normed 格式
            hr_imgs = hr_imgs.to(device)  # (batch_size (N), 3, 96, 96),  [-1, 1]格式

            # 前向传播
            sr_imgs = model(lr_imgs)

            # 计算损失
            result_loss = loss(sr_imgs, hr_imgs)  

            # 后向传播
            optimizer.zero_grad()
            result_loss.backward()

            # 更新模型
            optimizer.step()
            all_loss += result_loss.item()*lr_imgs.size()[0]

        # 手动释放内存              
        del lr_imgs, hr_imgs, sr_imgs

        print('Train Epoch: {} \tLoss: {:.6f}\n'.format(
            epoch,
            all_loss / len(train_dataloader.dataset))
        )

    def evaluate(model, loss, test_dataloader, epoch):
        model.eval()
        test_loss = 0
        with torch.no_grad():
            for i, (lr_imgs, hr_imgs) in enumerate(test_dataloader):
                lr_imgs = lr_imgs.to(device)  # (batch_size (N), 3, 24, 24), imagenet-normed 格式
                hr_imgs = hr_imgs.to(device)  # (batch_size (N), 3, 96, 96),  [-1, 1]格式
                sr_imgs = model(lr_imgs)
                test_loss += loss(sr_imgs, hr_imgs).item()*lr_imgs.size()[0]
                
        test_loss /= len(test_dataloader.dataset)
        fmt = '\nValidation set: Loss: {:.4f}\n'
        print(
            fmt.format(
                test_loss,
            )
        )

        return test_loss

    def train_and_evaluate(model, loss_function, train_dataloader, val_dataloader, optimizer, epochs, save_path):
        current_min_test_loss = 100
        for epoch in range(1, epochs + 1):
            train(model, loss_function, train_dataloader, optimizer, epoch)
            test_loss = evaluate(model, loss_function, val_dataloader, epoch)
            if test_loss < current_min_test_loss:
                print("The validation loss is improved from {} to {}, new model weight is saved.".format(
                    current_min_test_loss, test_loss))
                current_min_test_loss = test_loss
                torch.save(model, save_path)
            else:
                print("The validation loss is not improved.")
            print("------------------------------------------------")

    train_and_evaluate(model, 
        loss_function=criterion, 
        train_dataloader=train_loader, 
        val_dataloader=val_loader, 
        optimizer=optimizer, 
        epochs=conf.epochs, 
        save_path=modelweightfile)

if __name__ == '__main__':
    main()
