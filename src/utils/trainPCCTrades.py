import time, os, signal

import torch as tc
import torch.nn as nn
from torchvision.transforms.functional import to_pil_image
from torch.utils.data import DataLoader
import torch.utils.data.distributed as dd
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.autograd import detect_anomaly, grad

from src.data.factory import fetch_dataset, DATASETS
from src.model.factory import *
from src.utils.helper import *
from src.utils.evaluate import validate
from src.utils.printer import sprint, dprint
from src.utils.adversary import pgd, perturb,tradeattack
from src.utils.swa import moving_average, bn_update

import numpy as np
import torch.nn.functional as F
from torch.autograd import Variable

def train1(args):
    start_epoch = 0
    best_acc1, best_pgd, best_fgsm = 0, 0, 0
    checkpoint = None
    train_set,  val_set = get_datasets(args)#tinyimagenet数据集
    train_sampler = dd.DistributedSampler(train_set) if args.parallel else None
    train_loader = DataLoader(train_set,
                              batch_size=args.batch_size,
                              shuffle=(train_sampler is None),
                              pin_memory=True,
                              num_workers=args.num_workers,
                              sampler=train_sampler,
                              drop_last=True)
    if args.world_size > 1:
        val_set = val_set[args.rank]
    val_loader = DataLoader(val_set,
                            batch_size=args.batch_size,
                            num_workers=args.num_workers,
                            pin_memory=True)
    
    if args.resume is not None:
        resume_file = args.path('trained', "{}/{}_end".format(args.logbook, args.resume))
        if os.path.isfile(resume_file):
            checkpoint = tc.load(resume_file, map_location='cpu')
            best_acc1 = checkpoint['best_acc1']
            best_fgsm = checkpoint['best_fgsm']
            best_pgd = checkpoint['best_pgd']
            start_epoch = checkpoint['epoch']
        else:
            raise Exception("Resume point not exists: {}".format(args.resume))
        checkpoint = (args.resume, checkpoint)
        
    fargs = args.func_arguments(fetch_model, ARCHS, postfix='arch')
    if checkpoint is not None:
        fargs['checkpoint'] = checkpoint
    model = fetch_model(**fargs).to(args.device)
    if args.parallel:
        if args.using_cpu():
            model = DDP(model)
        else:
            model = DDP(model, device_ids=[args.rank], output_device=args.rank)
    #不用管：   
    if args.swa is not None:
        if args.resume is None or start_epoch <= args.swa_start:
            swa_model = fetch_model(**fargs).to(args.device)
            swa_best_acc = 0.0
            swa_best_fgm = 0.0
            swa_best_pgd = 0.0
            args.swa_n = 0
        else:
            swa_ckp = args.path('trained', "{}/{}_swa_end".format(args.logbook, args.resume))
            swa_ckp = tc.load(swa_ckp, map_location='cpu')
            fargs['checkpoint'] = (args.resume, swa_ckp)
            swa_model = fetch_model(**fargs).to(args.device)
            swa_best_acc = swa_ckp['best_acc']
            swa_best_fgm = swa_ckp['best_fgm']
            swa_best_pgd = swa_ckp['best_pgd']
            args.swa_n = swa_ckp['num']

        args.swa_freq = len(train_loader) if args.swa_freq == -1 else args.swa_freq
    else:
        swa_model = None
        
    criterion = nn.CrossEntropyLoss()

    fargs = args.func_arguments(fetch_optimizer, OPTIMS, checkpoint=checkpoint)
    optimizer = fetch_optimizer(params=model.parameters(), **fargs)
    
    if args.advt:
        dprint('adversary', **{k:getattr(args, k, None)
                               for k in ['eps', 'eps_step', 'max_iter', 'eval_iter']})
    dprint('data loader', batch_size=args.batch_size, num_workers=args.num_workers)
    sprint("=> Start training!", split=True)
    
    with open("TradesswaPCCD.txt", "w") as f:
   
     for epoch in range(start_epoch, args.epochs):
        if args.parallel:
            train_sampler.set_epoch(epoch)
            
        adjust_learning_rate(optimizer, epoch, args.lr, args.annealing)#[100-150]
        # adjust_learning_rate(optimizer, epoch, args.lr, args.epochs)
        
        update(train_loader, model, criterion, optimizer, epoch, args, swa_model)
        acc1, ig, fgsm, pgd = validate(val_loader, model, criterion, args)
        
        if args.rank is not None and args.rank != 0: continue
        # execute only on the main process
        
        best_acc1 = max(acc1, best_acc1)
        best_fgsm = max(fgsm, best_fgsm)
        best_pgd = max(pgd, best_pgd)
        # ig_norm = tc.norm(ig, p=1)
        # print(" ** Acc@1: {:.2f} | FGSM: {:.2f} | PGD: {:.2f}".format(best_acc1, best_fgsm, best_pgd))
        result_str = "Epoch: {} ** BAcc@1: {:.2f} | BFGSM: {:.2f} | BPGD: {:.2f} |ig:{:.2f}| Acc@1: {:.2f} | FGSM: {:.2f} | PGD: {:.2f} |\n".format(epoch, best_acc1, best_fgsm, best_pgd, ig, acc1, fgsm, pgd)
        print(result_str)
        f.write(result_str + " ")
        
        if args.logging:
            logger = args.logger
            acc_info = '{:3.2f} E: {} IG: {:.2e} FGSM: {:3.2f} PGD: {:3.2f}'.format(acc1, epoch+1, ig, fgsm, pgd )
            logger.update('checkpoint', end=acc_info)
            state_dict = model.module.state_dict() if args.parallel else model.state_dict()
            state = {
                'epoch': epoch + 1,
                'state_dict': state_dict,
                'best_acc1': best_acc1,
                'best_pgd': best_pgd,
                'best_fgsm' : best_fgsm,
                'optimizer' : optimizer.state_dict(),
            }

            if acc1 >= best_acc1:
                logger.update('checkpoint', acc=acc_info, save=True)

            lid = args.log_id
            fname = "{}/{}".format(args.logbook, lid)
            ck_path = args.path('trained', fname+"_end")
            tc.save(state, ck_path)

            if acc1 >= best_acc1:
                shutil.copyfile(ck_path, args.path('trained', fname+'_acc'))

            if pgd >= best_pgd:
                logger.update('checkpoint', pgd=acc_info, save=True)
                shutil.copyfile(ck_path, args.path('trained', fname+'_pgd'))

            if args.swa is not None and args.swa_start <= epoch:
                print(" *  averaging the model")
                bn_update(train_loader, swa_model)
                swa_model.eval()
                swa_acc, swa_ig, swa_fgm, swa_pgd = validate(val_loader, swa_model, criterion, args)
                
                swa_best_acc = max(swa_acc, swa_best_acc)
                swa_best_fgm = max(swa_fgm, swa_best_fgm)
                swa_best_pgd = max(swa_pgd, swa_best_pgd)
                
                print(" ** Acc@1: {:.2f} | FGSM: {:.2f} | PGD: {:.2f}".format(swa_best_acc,
                                                                              swa_best_fgm,
                                                                              swa_best_pgd))
                state = {'state_dict' : swa_model.state_dict(),
                         'num': args.swa_n,
                         'best_acc' : swa_best_acc,
                         'best_pgd' : swa_best_pgd,
                         'best_fgm' : swa_best_fgm}

                ck_path = args.path('trained', fname+"_swa_end")
                tc.save(state, ck_path)

                if swa_pgd >= swa_best_pgd:
                    shutil.copyfile(ck_path, args.path('trained', fname+'_swa_pgd'))
                        
# def SmoothLoss2(opt2, img,model,target):  
#     smoothing=0.3
#     log_prob = F.log_softmax(opt2, dim=-1)  
#     num_classes = log_prob.size(-1)  
#     # 创建平滑后的标签  
#     with torch.no_grad():  
#         # 如果target是类别索引，先转换为one-hot编码  
#         if target.dim() == 1:  
#             target = F.one_hot(target, num_classes=num_classes).float()  
#         smoothed_target = target * (1. - smoothing) + smoothing / num_classes     
#     prob_natural = F.softmax(smoothed_target, dim=1)  
#     loss_ce = (F.cross_entropy(log_prob, torch.argmax(prob_natural, dim=1))+ + grad_align_loss(model, img, target)).mean()#正则
#     return loss_ce    

def PCC_Loss(opt2, img,model,target):  
    smoothing=0.1
    log_prob = F.log_softmax(opt2, dim=-1)  
    num_classes = log_prob.size(-1)  
    with tc.no_grad():  
        # 如果target是类别索引，先转换为one-hot编码  
        if target.dim() == 1:  
            target = F.one_hot(target, num_classes=num_classes).float()  
        smoothed_target = target * (1. - smoothing) + smoothing / num_classes     
    # 计算标签平滑损失  
    # loss = (-smoothed_target * log_prob).sum(dim=-1).mean()##不正则
    loss = ((-smoothed_target * log_prob).sum(dim=-1) + grad_align_loss(model, img, target)).mean()#正则
    return loss    

#元损失
import math
import sys
inf = math.inf
nan = math.nan
string_classes = (str, bytes)
PY37 = sys.version_info[0] == 3 and sys.version_info[1] >= 7
def with_metaclass(meta: type, *bases) -> type:
    """Create a base class with a metaclass."""
    # This requires a bit of explanation: the basic idea is to make a dummy
    # metaclass for one level of class instantiation that replaces itself with
    # the actual metaclass.
    class metaclass(meta):  # type: ignore[misc, valid-type]

        def __new__(cls, name, this_bases, d):
            return meta(name, bases, d)

        @classmethod
        def __prepare__(cls, name, this_bases):
            return meta.__prepare__(name, bases)

    return type.__new__(metaclass, 'temporary_class', (), {})
class VariableMeta(type):
    def __instancecheck__(cls, other):
        return isinstance(other, torch.Tensor)
# mypy doesn't understand torch._six.with_metaclass
class Variable(with_metaclass(VariableMeta, tc._C._LegacyVariableBase)):  # type: ignore[misc]
    pass
def LabelSmoothLoss(input, target):
    log_prob = F.log_softmax(input, dim=-1)
    loss = (-target * log_prob).sum(dim=-1).mean()
    return loss
def _label_smoothing(label,factor):
    factor=0.7
    one_hot = np.eye(10)[label.cuda().data.cpu().numpy()]
    result = one_hot * factor + (one_hot - 1.) * ((factor - 1) / float(100 - 1))
    return result

def update(loader, model, criterion, optimizer, epoch, args, swa_model):
    batch_time = AverageMeter('Time', ':6.3f')
    losses = AverageMeter('Loss', ':.2e')
    igs = AverageMeter('IG', ':.2e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    meters = [batch_time, losses, igs, top1, top5]
    progress = ProgressMeter(len(loader), meters, prefix="Epoch: [{}]".format(epoch))
    model.train()
    end = time.time()
    
    for i, (img, tgt) in enumerate(loader, 1):
        img = img.to(args.device, non_blocking=True)
        tgt = tgt.to(args.device, non_blocking=True)        
        
        batch_size = len(img)

        if args.warm_start and epoch < 5:
            factor = epoch / 5
            eps = args.eps * factor
            step = args.eps_step * factor
        else:
            eps, step = args.eps, args.eps_step
            
        img.requires_grad_(True)
        opt = model(img)
        loss = criterion(opt, tgt)
        
        ig = grad(loss, img)[0]
        ig_norm = tc.norm(ig, p=1) 
        # adv, prt = pgd(img, tgt, model, criterion, eps, step, args.max_iter, ig)
        adv = tradeattack(model,img,step,eps,args.max_iter)
        
        opt2 = model(adv)
        loss = PCC_Loss(opt2, img,model,tgt)
        # loss = F.kl_div(F.log_softmax(opt2, dim=1),
        #                        F.softmax(opt, dim=1),
        #                        reduction='batchmean')
        
        acc1, acc5 = accuracy(opt, tgt, topk=(1, 5))
        top1.update(acc1[0], batch_size)
        top5.update(acc5[0], batch_size)
        losses.update(loss.item(), batch_size)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        igs.update(ig_norm, batch_size)
        
        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()
        
        if i == 1 or i % args.log_pbtc == 0:
            progress.display(i)
        
        if args.rank != 0: continue
        
        if args.swa is not None and args.swa_start <= epoch and i % args.swa_freq == 0:
            if isinstance(args.swa_decay, str):
                moving_average(swa_model, model, 1.0 / (args.swa_n + 1))
                args.swa_n += 1
            else:
                if epoch == args.swa_start and i // args.swa_freq == 1:
                    state_dict = model.module.state_dict() if args.parallel else model.state_dict()
                    swa_model.load_state_dict(state_dict)
                moving_average(swa_model, model, args.swa_decay)

def adjust_learning_rate(optimizer, epoch, lr, annealing):
    decay = 0
    for a in annealing:
        if epoch < int(a): break
        else: decay += 1
    lr *= 0.1 ** decay
    params = optimizer.param_groups
    if lr != params[0]['lr']:
        print("Learning rate now is {:.0e}".format(lr))
    for param in params: param['lr'] = lr
    print("Epoch [{}], Learning rate now is {:.4f}".format(epoch, lr))  

# def adjust_learning_rate(optimizer, epoch, lr, total_epochs):  
#     # 计算余弦退火中的lambda值  
#     T_max = int(total_epochs * 0.8) 
#     if epoch < T_max:  
#         lr_factor = 0.5 * (1 + np.cos(np.pi * epoch / T_max))  
#     else:  
#         lr_factor = 0.001 # 或者保持最后的学习率不变  
#     # 更新学习率  
#     new_lr = lr * lr_factor  
#     for param_group in optimizer.param_groups:  
#         param_group['lr'] = new_lr  
#     # 打印新的学习率（可选）  
#     print("Epoch [{}/{}], Learning rate now is {:.4f}".format(epoch, total_epochs, new_lr))  
    
def grad_align_loss(model, X, y):
    # grad1 = get_input_grad(model, X, y,  args.train_eps, delta_init='random_corner', backprop=False)
    grad1 = get_input_grad(model, X, y,  eps=0.031, delta_init='none', backprop=False)
    grad2 = get_input_grad(model, X, y,   eps=0.031, delta_init='random_corner', backprop=True)
    grad1, grad2 = grad1.reshape(len(grad1), -1), grad2.reshape(len(grad2), -1)
    grad1_normed = F.normalize(grad1, dim=1, eps=1e-6)  
    grad2_normed = F.normalize(grad2, dim=1, eps=1e-6)  
    reg = 0.2 * tc.mean(grad1_normed * grad2_normed, dim=1)
    return reg

def get_uniform_delta(shape, eps, requires_grad=True):
    delta = tc.zeros(shape).cuda()
    delta.uniform_(-eps, eps)
    delta.requires_grad = requires_grad
    return delta
def Loss(opt2,target):  
    smoothing=0.1
    log_prob = F.log_softmax(opt2, dim=-1)  
    num_classes = log_prob.size(-1)  
    # 创建平滑后的标签  
    with tc.no_grad():  
        # 如果target是类别索引，先转换为one-hot编码  
        if target.dim() == 1:  
            target = F.one_hot(target, num_classes=num_classes).float()  
        smoothed_target = target * (1. - smoothing) + smoothing / num_classes     
    # 计算标签平滑损失  
    loss = (-smoothed_target * log_prob).sum(dim=-1).mean()
    return loss

def get_input_grad(model, X, y, eps, delta_init='none', backprop=False):
    if delta_init == 'none':
        delta = tc.zeros_like(X, requires_grad=True)
    # print( lower_limit.shape)
    # print( torch.sign(grad).shape)
    elif delta_init == 'random_uniform':
        delta = get_uniform_delta(X.shape, eps, requires_grad=True)
    elif delta_init == 'random_corner':
        delta = get_uniform_delta(X.shape, eps, requires_grad=True)
        delta = eps * tc.sign(delta)
    else:
        raise ValueError('wrong delta init')

    output = model(X + delta)  
    loss_per_sample = Loss(output, y)  # 注意这里只传入 output 和 y  
    loss = loss_per_sample.mean()  # 计算平均损失  
  
    grad = tc.autograd.grad(loss, delta, create_graph=True if backprop else False)[0]  
      
    if not backprop:  
        grad, delta = grad.detach(), delta.detach()  
    return grad  

import shutil
import time
import sys
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
# from torch.autograd.gradcheck import zero_gradients
import torch.nn.functional as F
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import DataLoader, Subset
from torchvision.datasets import CIFAR10, CIFAR100, ImageFolder,SVHN
from torch.autograd import Variable
import torch.autograd as autograd
def zero_gradients(inputs):
    if isinstance(inputs, Variable):
        if inputs.grad is not None:
            inputs.grad.data.zero_()
import numpy as np
################################ datasets #######################################
def cifar10_dataloaders(batch_size=128, num_workers=2, data_dir='/root/WYH/project/data/cifar10'):

    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
    ])

    test_transform = transforms.Compose([
        transforms.ToTensor(),
    ])

    train_set = Subset(CIFAR10(data_dir, train=True, transform=train_transform, download=True), list(range(45000)))
    val_set = Subset(CIFAR10(data_dir, train=True, transform=test_transform, download=True), list(range(45000, 50000)))
    test_set = CIFAR10(data_dir, train=False, transform=test_transform, download=True)

    return train_set, test_set

def cifar100_dataloaders(batch_size=128, num_workers=2, data_dir='/media/disk/wyh/MYproject/robust/pytorch/data/CIFAR100'):

    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ToTensor(),
    ])

    test_transform = transforms.Compose([
        transforms.ToTensor(),
    ])

    train_set = Subset(CIFAR100(data_dir, train=True, transform=train_transform, download=True), list(range(45000)))
    val_set = Subset(CIFAR100(data_dir, train=True, transform=test_transform, download=True), list(range(45000, 50000)))
    test_set = CIFAR100(data_dir, train=False, transform=test_transform, download=True)

    return train_set, test_set
def svhn_dataloaders(batch_size=128,num_workers=2, data_dir = '/media/disk/wyh/MYproject/robust/pytorch/data/svhn'):

    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ToTensor(),
    ])

    test_transform = transforms.Compose([
        transforms.ToTensor(),
    ])

    train_set = Subset(SVHN(data_dir, split='train', transform=train_transform, download=True),list(range(68257)))
    val_set = Subset(SVHN(data_dir, split='train', transform=train_transform, download=True),list(range(68257,73257)))
    test_set = SVHN(data_dir, split='test', transform=test_transform, download=True)
            

    return train_set, test_set

import os
from PIL import Image
from torch.utils.data import Dataset

class NWPU_RESISC45(Dataset):
    def __init__(self, root_dir, train=True, transform=None, download=False):
        self.root_dir = root_dir
        self.train = train
        self.transform = transform
        self.samples = self._load_samples()

    def _load_samples(self):
        if self.train:
            subfolder = 'train'
        else:
            subfolder = 'test'

        samples = []
        for class_name in os.listdir(os.path.join(self.root_dir, subfolder)):
            class_dir = os.path.join(self.root_dir, subfolder, class_name)
            for file_name in os.listdir(class_dir):
                file_path = os.path.join(class_dir, file_name)
                samples.append((file_path, class_name))

        return samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        if idx >= len(self.samples):
            raise IndexError("list index out of range")

        path, target = self.samples[idx]
        image = Image.open(path).convert('RGB')

        if self.transform:
            image = self.transform(image)

        return image, target
def nwpu_resisc45_dataloaders(batch_size=128, num_workers=2, data_dir='/media/disk/wyh/MYproject/robust/pytorch/data/NWPU-RESISC45'):
    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
    ])

    test_transform = transforms.Compose([
        transforms.ToTensor(),
    ])

    train_set = Subset(NWPU_RESISC45(data_dir, train=True, transform=train_transform, download=True), list(range(4500)))
    val_set = Subset(NWPU_RESISC45(data_dir, train=True, transform=test_transform, download=True), list(range(4500, 5000)))
    test_set = NWPU_RESISC45(data_dir, train=False, transform=test_transform, download=True)
    
    return train_set, test_set

from PIL import Image
import tifffile as tiff

class SIRI_Dataset(Dataset):
    def __init__(self, root_dir, train=True, transform=None, download=False):
        self.root_dir = root_dir
        self.train = train
        self.transform = transform
        self.samples = self._load_samples()

    def _load_samples(self):
        if self.train:
            subfolder = 'train'
        else:
            subfolder = 'test'

        samples = []
        for class_name in os.listdir(os.path.join(self.root_dir, subfolder)):
            class_dir = os.path.join(self.root_dir, subfolder, class_name)
            for file_name in os.listdir(class_dir):
                file_path = os.path.join(class_dir, file_name)
                samples.append((file_path, class_name))

        return samples

    def __len__(self):
        return len(self.samples)
    def __getitem__(self, idx):
        if idx >= len(self.samples):
            raise IndexError("list index out of range")
        path, target = self.samples[idx]
        image = tiff.imread(path).transpose([1, 2, 0])  # 使用tifffile库读取.tif文件，并转置通道
        image = Image.fromarray(image)  # 将numpy数组转换为PIL图像
        if self.transform:
            image = self.transform(image)
        return image, target

def siri_dataloaders(batch_size=128, num_workers=2, data_dir='/media/disk/wyh/MYproject/robust/pytorch/data/SIRI'):

    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
    ])

    test_transform = transforms.Compose([
        transforms.ToTensor(),
    ])

    train_set = Subset(SIRI_Dataset(data_dir, train=True, transform=train_transform, download=True), list(range(4500)))
    val_set = Subset(SIRI_Dataset(data_dir, train=True, transform=test_transform, download=True), list(range(4500, 5000)))
    test_set = SIRI_Dataset(data_dir, train=False, transform=test_transform, download=True)

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)

    return train_loader, test_loader

def get_datasets(args):
    if args.datasets == 'CIFAR10':
        return cifar10_dataloaders(batch_size=args.batch_size, num_workers=args.workers)

    elif args.datasets == 'CIFAR100':
        return cifar100_dataloaders(batch_size=args.batch_size, num_workers=args.workers)

    elif args.datasets == 'TinyImagenet':
        return tiny_imagenet_dataloaders(batch_size=args.batch_size, num_workers=args.workers)
    elif args.datasets == 'SVHN':
        return svhn_dataloaders(batch_size=args.batch_size, num_workers=args.workers)
    elif args.datasets == 'NWPU':
        return nwpu_resisc45_dataloaders(batch_size=args.batch_size, num_workers=args.workers)

from PIL import Image  
from torch.utils.data import Dataset
from torchvision import transforms  

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
        transforms.Resize((32, 32)),  # 调整图像大小为32x32  
        transforms.RandomHorizontalFlip(),  
        transforms.ToTensor(),  
        # transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),  # 归一化  
    ])  
  
    val_transform = transforms.Compose([  
        transforms.Resize((32, 32)),  # 调整图像大小为32x32  
        transforms.ToTensor(),  
        # transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),  # 归一化  
    ])   
    test_transform = val_transform  # 验证集和测试集使用相同的预处理  
  # 加载数据集
    train_set = NWPU_RESISC45(root_dir='/media/disk/wyh/MYproject/robust/pytorch/data/NWPU/train', train=True, transform=train_transform)
# 获取训练集的长度
    train_set_length = len(train_set)
# 确保验证集的索引范围在有效范围内
    val_indices = list(range(22050, min(25950, train_set_length)))
# 创建验证集
    val_set = Subset(train_set, val_indices)
# 加载测试集
    test_set = NWPU_RESISC45(root_dir='/media/disk/wyh/MYproject/robust/pytorch/data/NWPU/test', train=False, transform=test_transform)
# 创建数据加载器
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return train_loader, val_loader, test_loader

from PIL import Image

def tiny_imagenet_dataloaders(batch_size=64, num_workers=8, data_dir='/media/disk/wyh/MYproject/robust/pytorch/data/tiny-imagenet-200'):
    transform_train = transforms.Compose([
        transforms.Lambda(lambda x: x.convert("RGB")),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    transform_test = transforms.Compose([
        transforms.Lambda(lambda x: x.convert("RGB")),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    trainset = TinyImageNet(data_dir, 'train', transform=transform_train, in_memory=True)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=8)

    testset = TinyImageNet(data_dir, 'val', transform=transform_test, in_memory=True)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=8)

    return trainloader, testloader


import glob
from torch.utils.data import Dataset

EXTENSION = 'JPEG'
NUM_IMAGES_PER_CLASS = 500
CLASS_LIST_FILE = '/media/disk/wyh/MYproject/robust/pytorch/data/tiny-imagenet-200/wnids.txt'
VAL_ANNOTATION_FILE = '/media/disk/wyh/MYproject/robust/pytorch/data/tiny-imagenet-200/val/val_annotations.txt'


class TinyImageNet(Dataset):
    """Tiny ImageNet data set available from `http://cs231n.stanford.edu/tiny-imagenet-200.zip`.
    Parameters
    ----------
    root: string
        Root directory including `train`, `test` and `val` subdirectories.
    split: string
        Indicating which split to return as a data set.
        Valid option: [`train`, `test`, `val`]
    transform: torchvision.transforms
        A (series) of valid transformation(s).
    in_memory: bool
        Set to True if there is enough memory (about 5G) and want to minimize disk IO overhead.
    """
    def __init__(self, root, split='train', transform=None, target_transform=None, in_memory=False):
        self.root = os.path.expanduser(root)
        self.split = split
        self.transform = transform
        self.target_transform = target_transform
        self.in_memory = in_memory
        self.split_dir = os.path.join(root, self.split)
        self.image_paths = sorted(glob.iglob(os.path.join(self.split_dir, '**', '*.%s' % EXTENSION), recursive=True))
        self.labels = {}  # fname - label number mapping
        self.images = []  # used for in-memory processing

        # build class label - number mapping
        with open(os.path.join(self.root, CLASS_LIST_FILE), 'r') as fp:
            self.label_texts = sorted([text.strip() for text in fp.readlines()])
        self.label_text_to_number = {text: i for i, text in enumerate(self.label_texts)}

        if self.split == 'train':
            for label_text, i in self.label_text_to_number.items():
                for cnt in range(NUM_IMAGES_PER_CLASS):
                    self.labels['%s_%d.%s' % (label_text, cnt, EXTENSION)] = i
        elif self.split == 'val':
            with open(os.path.join(self.split_dir, VAL_ANNOTATION_FILE), 'r') as fp:
                for line in fp.readlines():
                    terms = line.split('\t')
                    file_name, label_text = terms[0], terms[1]
                    self.labels[file_name] = self.label_text_to_number[label_text]

        # read all images into torch tensor in memory to minimize disk IO overhead
        if self.in_memory:
            self.images = [self.read_image(path) for path in self.image_paths]

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, index):
        file_path = self.image_paths[index]

        if self.in_memory:
            img = self.images[index]
        else:
            img = self.read_image(file_path)

        if self.split == 'test':
            return img
        else:
            # file_name = file_path.split('/')[-1]
            return img, self.labels[os.path.basename(file_path)]

    def __repr__(self):
        fmt_str = 'Dataset ' + self.__class__.__name__ + '\n'
        fmt_str += '    Number of datapoints: {}\n'.format(self.__len__())
        tmp = self.split
        fmt_str += '    Split: {}\n'.format(tmp)
        fmt_str += '    Root Location: {}\n'.format(self.root)
        tmp = '    Transforms (if any): '
        fmt_str += '{0}{1}\n'.format(tmp, self.transform.__repr__().replace('\n', '\n' + ' ' * len(tmp)))
        tmp = '    Target Transforms (if any): '
        fmt_str += '{0}{1}'.format(tmp, self.target_transform.__repr__().replace('\n', '\n' + ' ' * len(tmp)))
        return fmt_str

    def read_image(self, path):
        img = Image.open(path)
        return self.transform(img) if self.transform else img

  