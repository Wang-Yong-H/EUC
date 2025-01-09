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

def adjust_learning_rate(optimizer, epoch, lr, total_epochs):  
    # 计算余弦退火中的lambda值  
      T_max = int(total_epochs * 0.8) 
      if epoch < T_max:  
        lr_factor = 0.5 * (1 + np.cos(np.pi * epoch / T_max))  
      else:  
        lr_factor = 0.001 # 或者保持最后的学习率不变  
    # 更新学习率  
      new_lr = lr * lr_factor  
      for param_group in optimizer.param_groups:  
        param_group['lr'] = new_lr  
    # 打印新的学习率（可选）  
      print("Epoch [{}/{}], Learning rate now is {:.4f}".format(epoch, total_epochs, new_lr))  
      
# 设置保存文件夹  
save_folder = os.path.join('results', 'NWPU/resnet50_training')  
os.makedirs(save_folder, exist_ok=True)  # 使用 os.makedirs 以确保文件夹存在  
  
# 加载 ResNet50 模型，这里设置为不使用预训练权重  
Ren50model = model.resnet50(weights=None)

if args.resume:  
    # 加载最佳验证模型（或根据需要加载训练模型）  
    model_path = os.path.join(save_folder, '/root/WYH/DA-Alone-Improves-AT-main/results/NWPU/resnet50_training/best_train_model.pth')  # 或 'best_train_model.pth'  
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
  
# 设置优化器  
optimizer = SGD(Ren50model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.wd)  
import torch.utils.data.distributed as dd

# 假设 train_data 是已经设置好的 DataLoader  
train_data,  test_data = nwpu_resisc45_dataloaders(batch_size=args.batch_size, num_workers=args.cpus)
# train_set,  val_set = RS_images_2800_dataloaders(batch_size=args.batch_size, num_workers=args.workers)
train_data,  test_data = RS_images_2800_dataloaders(batch_size=args.batch_size, num_workers=args.cpus)
# 初始化训练曲线列表  
training_loss, val_TA = [], []  # 注意：这里 val_TA 可能需要在验证循环中填充  
# 训练循环 
# 初始化最佳训练精度和验证精度  
best_train_acc = 0   
best_val_acc = 0  
 
for epoch in range(args.epochs):  
    Ren50model.train()  
    total_loss = 0  
    correct = 0  
    total = 0  
    adjust_learning_rate(optimizer, epoch, args.lr, args.epochs)
    for i, (imgs, labels) in enumerate(tqdm(train_data, desc=f'Epoch {epoch+1}/{args.epochs}', leave=False)):  
        imgs, labels = imgs.cuda(), labels.cuda()  
        
        # 前向传播  
        outputs = Ren50model(imgs)  
        loss = F.cross_entropy(outputs, labels)  
  
        # 反向传播和优化  
        optimizer.zero_grad()  
        loss.backward()  
        optimizer.step()  
  
        # 计算准确率  
        _, predicted = torch.max(outputs.data, 1)  
        total += labels.size(0)  
        correct += (predicted == labels).sum().item()  
        total_loss += loss.item()  
  
     # 计算平均损失和准确率  
    avg_loss = total_loss / len(train_data)  
    train_acc = 100. * correct / total  
  
    # 验证阶段  
    Ren50model.eval()  # 设置模型为评估模式  
    val_loss = 0  
    val_correct = 0  
    val_total = 0  
    with torch.no_grad():  # 禁用梯度计算  
        for imgs, labels in test_data:  
            imgs, labels = imgs.cuda(), labels.cuda()  
            outputs = Ren50model(imgs)  
            loss = F.cross_entropy(outputs, labels)  
            val_loss += loss.item()  
            _, predicted = torch.max(outputs.data, 1)  
            val_total += labels.size(0)  
            val_correct += (predicted == labels).sum().item()  
  
    # 计算验证平均损失和准确率  
    val_avg_loss = val_loss / len(test_data)  
    val_acc = 100. * val_correct / val_total  
  
    # 保存训练日志  
    with open(os.path.join(save_folder, 'train_log.txt'), 'a+') as f:  
        f.write(f'Epoch {epoch+1}, Train Loss: {avg_loss:.4f}, Train Accuracy: {train_acc:.2f}%, Val Loss: {val_avg_loss:.4f}, Val Accuracy: {val_acc:.2f}%\n')  
  
    # 更新最佳精度并保存模型  
    if train_acc > best_train_acc:  
        best_train_acc = train_acc  
        torch.save(Ren50model.state_dict(), os.path.join(save_folder, 'best_train_model.pth'))  
    if val_acc > best_val_acc:  
        best_val_acc = val_acc  
        torch.save(Ren50model.state_dict(), os.path.join(save_folder, 'best_val_model.pth'))  
    print(f'Epoch {epoch}, Train Loss: {avg_loss:.4f}, Train Accuracy: {train_acc:.2f}%, Val Loss: {val_avg_loss:.4f}, Val Accuracy: {val_acc:.2f}%\n')   

