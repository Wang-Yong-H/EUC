import torch  
import torchvision.transforms as transforms  
from PIL import Image  
import os  

# '''
# PGDAT
# '''

import os, argparse, time
import numpy as np
# import matplotlib.pyplot as plt

import torch
from tqdm import tqdm

import torch.nn.functional as F
from torch.optim import SGD, Adam, lr_scheduler
import torchvision
import torchvision.models as model  
from torchvision import datasets, transforms  
from torch.utils.data import DataLoader, SubsetRandomSampler     
import os
from PIL import Image
from torch.utils.data import Dataset
from torch.utils.data import DataLoader, Subset 

parser = argparse.ArgumentParser(description='CIFAR10 Training against DDN Attack')
parser.add_argument('--gpu', default='0')
parser.add_argument('--cpus', default=4)
# dataset:
parser.add_argument('--dataset', '--ds', default='attack', choices=['cifar10', 'svhn', 'stl10','attack'], help='which dataset to use')
# optimization parameters:
parser.add_argument('--batch_size', '-b', default=32, type=int, help='mini-batch size')
parser.add_argument('--epochs', '-e', default=200, type=int, help='number of total epochs to run')
parser.add_argument('--decay_epochs', '--de', default=[50,150], nargs='+', type=int, help='milestones for multisteps lr decay')
parser.add_argument('--opt', default='sgd', choices=['sgd', 'adam'], help='which optimizer to use')
parser.add_argument('--decay', default='cos', choices=['cos', 'multisteps'], help='which lr decay method to use')
parser.add_argument('--lr', default=0.1, type=float, help='initial learning rate')
parser.add_argument('--momentum', default=0.9, type=float, help='momentum')
parser.add_argument('--wd', default=5e-4, type=float, help='weight decay')
# adv parameters:
parser.add_argument('--targeted', action='store_true', help='If true, targeted attack')
parser.add_argument('--eps', type=int, default=8)
parser.add_argument('--steps', type=int, default=7)
# loss parameters:
parser.add_argument('--Lambda', default=0.5, type=float, help='adv loss tradeoff parameter')
# others:
parser.add_argument('--resume', action='store_true', help='If true, resume from early stopped ckpt')
args = parser.parse_args()
# print(args)

class NWPU_RESISC45(Dataset):  
    def __init__(self, root_dir, train=True, transform=None, target_transform=None):  
        """  
        Args:  
            root_dir (string): Directory with all the images and subfolders.  
            train (bool, optional): If True, creates dataset from training set, otherwise  
                creates from test set.  
            transform (callable, optional): Optional transform to be applied  
                on a sample.  
            target_transform (callable, optional): Optional transform to be applied on a target.  
        """  
        self.root_dir = root_dir  
        self.transform = transform  
        self.target_transform = target_transform  
        self.train = train  
          
        # 获取所有类别的名字  
        self.classes = os.listdir(root_dir)  
        if self.train:  
            self.classes = [cls for cls in self.classes if os.path.isdir(os.path.join(root_dir, cls))]  
            self.class_to_idx = {cls_name: i for i, cls_name in enumerate(self.classes)}  
        else:  
            # 对于测试集，你可能需要另外处理，因为测试集的结构可能与训练集不同  
            # 这里我们假设测试集也是类似的结构  
            self.class_to_idx = {cls_name: i for i, cls_name in enumerate(self.classes)}  
          
        # 构建样本和标签的列表  
        self.samples = []  
        for cls in self.classes:  
            class_dir = os.path.join(self.root_dir, cls)  
            if os.path.isdir(class_dir):  
                for file in os.listdir(class_dir):  
                    if file.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):  
                        path = os.path.join(class_dir, file)  
                        self.samples.append((path, self.class_to_idx[cls]))  

    def __len__(self):  
        return len(self.samples)  
  
    def __getitem__(self, idx):

      path, target = self.samples[idx]
      image = Image.open(path).convert('RGB')

      if self.transform:
         image = self.transform(image)

      return image, target
def nwpu_resisc45_dataloaders(batch_size=64, num_workers=2):  
    # 数据预处理 
    train_transform = transforms.Compose([  
        transforms.Resize((256, 256)),  # 可选，根据需要调整图片大小  
        transforms.RandomCrop(224),     # 裁剪到224x224，如果不需要随机裁剪可以去掉  
        transforms.RandomHorizontalFlip(),  
        transforms.ToTensor(),  
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),  # ImageNet的标准化参数  
    ])  
    val_transform = transforms.Compose([  
        transforms.Resize((256, 256)),  # 可选，根据需要调整图片大小  
        transforms.RandomCrop(224),     # 裁剪到224x224，如果不需要随机裁剪可以去掉  
        transforms.ToTensor(),  
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),  # ImageNet的标准化参数  
    ])  
        
    test_transform = val_transform  # 验证集和测试集使用相同的预处理  
  # 加载数据集
    train_set = NWPU_RESISC45(root_dir='/root/WYH/data/NWPU/train', train=True, transform=train_transform)
# 获取训练集的长度
    train_set_length = len(train_set)
# 确保验证集的索引范围在有效范围内
    val_indices = list(range(22050, min(25950, train_set_length)))
# 创建验证集
    val_set = Subset(train_set, val_indices)
# 加载测试集
    test_set = NWPU_RESISC45(root_dir='/root/WYH/data/NWPU/test', train=False, transform=test_transform)
# 创建数据加载器
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    # val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    return train_loader, test_loader

def RS_images_2800_dataloaders(batch_size=64, num_workers=2):  
    # 数据预处理 
    train_transform = transforms.Compose([  
        transforms.Resize((256, 256)),  # 可选，根据需要调整图片大小  
        transforms.RandomCrop(224),     # 裁剪到224x224，如果不需要随机裁剪可以去掉  
        transforms.RandomHorizontalFlip(),  
        transforms.ToTensor(),  
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),  # ImageNet的标准化参数  
    ])  
    val_transform = transforms.Compose([  
        transforms.Resize((256, 256)),  # 可选，根据需要调整图片大小  
        transforms.RandomCrop(224),     # 裁剪到224x224，如果不需要随机裁剪可以去掉  
        transforms.ToTensor(),  
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),  # ImageNet的标准化参数  
    ])  
        
    test_transform = val_transform  # 验证集和测试集使用相同的预处理  
  # 加载数据集
    train_set = NWPU_RESISC45(root_dir='/root/WYH/data/RSSCN7/train', train=True, transform=train_transform)
# 获取训练集的长度
    train_set_length = len(train_set)
# 确保验证集的索引范围在有效范围内
    val_indices = list(range(22050, min(25950, train_set_length)))
# 创建验证集
    val_set = Subset(train_set, val_indices)
# 加载测试集
    test_set = NWPU_RESISC45(root_dir='/root/WYH/data/RSSCN7/test', train=False, transform=test_transform)
# 创建数据加载器
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    # val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    return train_loader, test_loader

def pgd_attack(model, criterion, img, target, eps, step_size, num_steps, device='cuda'):  
    """  
    执行PGD攻击以生成对抗性样本。  
  
    参数:  
    - model: 加载好的模型  
    - criterion: 损失函数  
    - img: 原始图像tensor  
    - target: 目标标签  
    - eps: 允许的最大扰动大小  
    - step_size: 梯度下降的步长  
    - num_steps: 迭代次数  
    - device: 模型和数据的设备（'cuda' 或 'cpu'）  
  
    返回:  
    - adv_img: 生成的对抗性样本  
    """  
    img = img.clone().detach().requires_grad_(True).to(device)  
    optimizer = torch.optim.SGD([img], lr=step_size)  
  
    for _ in range(num_steps):  
        optimizer.zero_grad()  
          
        # 前向传播  
        outputs = model(img)  
        loss = criterion(outputs, target)  
          
        # 反向传播  
        loss.backward()  
          
        # 梯度符号  
        grad_sign = img.grad.data.sign()  
          
        # 更新图像（投影梯度下降）  
        perturbed_data = img + step_size * grad_sign  
          
        # 裁剪扰动以保持在epsilon球内  
        eta = torch.clamp(perturbed_data - img, min=-eps, max=eps)  
        img = img.detach() + eta  
        img = torch.clamp(img, 0, 1)  # 保持图像数据在[0, 1]范围内  
  
        img.requires_grad_(True)  
  
    return img  
  
def save_adversarial_images(adv_imgs, epoch, data_loader, save_dir='adv_images'):  
    """  
    保存对抗性图像到指定目录。  
  
    参数:  
    - adv_imgs: 对抗性图像tensor列表  
    - epoch: 当前训练轮次  
    - batch_idx: 当前batch索引  
    - data_loader: 数据加载器（用于获取文件名或索引）  
    - save_dir: 保存图像的目录  
    """  
    if not os.path.exists(save_dir):  
        os.makedirs(save_dir)  
  
    for i, adv_img in enumerate(adv_imgs):  
        # 假设data_loader的dataset有__getitem__可以返回文件名或索引  
        # 这里我们假设data_loader.dataset是一个简单的数据集，其中每个元素都是(image, label)  
        # 实际应用中，你可能需要根据你的数据集进行适当的修改  
        # 注意：这里为了简化，我们直接使用索引作为文件名的一部分  
        file_name = f'adv_img_{epoch}_{i}.png'  
        save_path = os.path.join(save_dir, file_name)  
  
        # 将tensor转换为PIL图像并保存  
        adv_img_pil = transforms.ToPILImage()(adv_img.cpu())  
        adv_img_pil.save(save_path)  
      
# 设置保存文件夹  
save_folder = '/root/WYH/DA-Alone-Improves-AT-main/results/NWPU/resnet50_training'
os.makedirs(save_folder, exist_ok=True)  # 使用 os.makedirs 以确保文件夹存在  
  
# 加载 ResNet50 模型，这里设置为不使用预训练权重  
Ren50model = model.resnet50(weights=None)

model_path = os.path.join(save_folder, '/best_train_model.pth')  # 或 'best_train_model.pth'  
if os.path.exists(model_path):  
        print(f"Loading model from checkpoint: {model_path}")  
        checkpoint = torch.load(model_path)  
        Ren50model.load_state_dict(checkpoint)
import torch.nn as nn       
# 检查 GPU 是否可用并设置 DataParallel  
if torch.cuda.is_available():  
    device_ids = list(range(torch.cuda.device_count()))  
    Ren50model = Ren50model.to(device_ids[0])  
    if len(device_ids) > 1:  
        Ren50model = nn.DataParallel(Ren50model, device_ids=device_ids) 
 
# 假设 train_data 是已经设置好的 DataLoader  
train_data,  test_data = nwpu_resisc45_dataloaders(batch_size=args.batch_size, num_workers=args.cpus)
# train_data,  test_data = RS_images_2800_dataloaders(batch_size=args.batch_size, num_workers=args.cpus)
Ren50model.eval()  # 设置模型为评估模式  
criterion = nn.CrossEntropyLoss()
         
for epoch, (images, targets) in enumerate(test_data):  
    # ...（其他代码，如模型训练）  
    adv_images = [pgd_attack(model, criterion, img.unsqueeze(0), target.unsqueeze(0), eps=0.03, step_size=0.01, num_steps=40, device='cuda') for img, target in zip(images, targets)]  
    save_adversarial_images(adv_images, epoch, test_data)  
 
