import torch as tc
from torch.autograd import grad

from torchattacks import *
import torch.nn.functional as F
from src.utils.printer import dprint
  
def input_grad2(imgs, targets, model, criterion):  
    output = model(imgs)  
    loss = criterion(output, targets)  
    ig = tc.autograd.grad(loss, imgs, retain_graph=True, create_graph=False)[0]  
    return ig  
  
def fgsm(imgs, targets, model, criterion, eps):  
    # 确保 imgs 需要梯度  
    imgs.requires_grad_(True)  
      
    # 计算梯度  
    ig = input_grad2(imgs, targets, model, criterion)      
    # 根据梯度生成扰动  
    pert = eps * tc.sign(ig)      
    # 约束扰动到 [-eps, eps] 范围内（虽然 FGSM 通常不需要这一步，因为 sign 函数输出已经是 -1 或 1）  
    # 但如果需要更精细的控制，可以保留  
    pert.clamp_(-eps, eps)     
    # 生成对抗样本  
    adv = tc.clamp(imgs + pert, 0, 1)  
    # 返回对抗样本和扰动  
    return adv.detach(), pert.detach() 
# adv_imgs, perturbations = fgsm(imgs, targets, model, criterion, eps)

def input_grad(imgs, targets, model, criterion):
    output = model(imgs)
    loss = criterion(output, targets)
    ig = grad(loss, imgs)[0]
    return ig

def grad1(loss_kl, x_adv):
    ig = grad(loss_kl, x_adv)[0]
    return ig

def perturb(imgs, targets, model, criterion, eps, eps_step, pert=None, ig=None):
    adv = imgs.requires_grad_(True) if pert is None else tc.clamp(imgs+pert, 0, 1).requires_grad_(True)
    ig = input_grad(adv, targets, model, criterion) if ig is None else ig
    if pert is None:
        pert = eps_step*tc.sign(ig)
    else:
        pert += eps_step*tc.sign(ig)
    pert.clamp_(-eps, eps)
    adv = tc.clamp(imgs+pert, 0, 1)
    pert = adv-imgs
    return adv.detach(), pert.detach()

def pgd(imgs, targets, model, criterion, eps, eps_step, max_iter, pert=None, ig=None):
    for i in range(max_iter):
        adv, pert = perturb(imgs, targets, model, criterion, eps, eps_step, pert, ig)
        ig = None
    return adv, pert

def ni_fgsm(imgs, targets, model, criterion, eps, mu=0.9, noise_std=0.1, num_noise=10):
    """
    执行NI-FGSM攻击。
    
    参数:
    - imgs: 输入图像张量。
    - targets: 目标标签张量。
    - model: 被攻击的模型。
    - criterion: 损失函数。
    - eps: 扰动大小。
    - mu: 噪声衰减因子（通常设为0.0）。
    - noise_std: 添加到输入图像上的高斯噪声的标准差。
    - num_noise: 要采样的噪声图像数量。
    
    返回:
    - adv: 对抗样本。
    - pert: 扰动。
    """
    # 将输入图像复制到GPU（如果可用）
    device = imgs.device
    
    # 初始化对抗样本和扰动
    adv = imgs.clone().detach().requires_grad_(True).to(device)
    pert = tc.zeros_like(adv).to(device)
    
    best_adv = adv.clone().detach()
    best_loss = float('inf')
    
    # 对每个噪声样本计算梯度并更新对抗样本
    for _ in range(num_noise):
        # 在输入图像上添加高斯噪声
        noise = tc.randn_like(imgs) * noise_std
        noisy_img = imgs + noise
        noisy_img = tc.clamp(noisy_img, 0, 1).to(device)
        
        # 计算噪声图像上的梯度
        grad = input_grad(noisy_img, targets, model, criterion)
        
        # 根据梯度更新扰动
        pert_step = eps * tc.sign(grad + mu * grad.mean(dim=(2, 3), keepdim=True))
        pert = tc.clamp(pert + pert_step, -eps, eps)
        
        # 应用扰动到原始输入图像上
        adv = tc.clamp(imgs + pert, 0, 1).detach().requires_grad_(False)
        
        # 计算当前对抗样本的损失
        outputs = model(adv)
        loss = criterion(outputs, targets)
        
        # 如果当前对抗样本的损失更小，则更新最佳对抗样本
        if loss < best_loss:
            best_loss = loss
            best_adv = adv.clone().detach()
    
    # 返回最佳对抗样本和对应的扰动
    return best_adv, best_adv - imgs

def tradeattack(model,
                  x_natural,
                  step_size=0.003,
                  epsilon=0.031,
                  perturb_steps=10):
      model.eval()
      x_adv = x_natural.detach() + 0.001 * tc.randn(x_natural.shape).to(x_natural.device).detach()
      for _ in range(perturb_steps):
            x_adv.requires_grad_()
            with tc.enable_grad():
                loss_kl = F.kl_div(F.log_softmax(model(x_adv), dim=1),
                                   F.softmax(model(x_natural), dim=1),
                                   reduction='sum')
            grad = grad1(loss_kl, x_adv)
            x_adv = x_adv.detach() + step_size * tc.sign(grad.detach())
            x_adv = tc.min(tc.max(x_adv, x_natural - epsilon), x_natural + epsilon)
            x_adv = tc.clamp(x_adv, 0.0, 1.0)
            pert = x_adv-x_natural.detach()
      return x_adv,pert
  
ATTACK = {
    'FGM' : FGSM,
    'PGD' : PGD,
    'APGD' : APGD,
    'AA' : AutoAttack,
    'AA+' : AutoAttack
}

HP_MAP = {
    'n_classes' : 'out_dim',
    'steps' : 'max_iter',
    'alpha' : 'eps_step',
    'n_restarts': 'num_random_init',
    'loss' : 'adv_loss',
    'random_start' : 'num_random_init',
    'norm' : 'adv_norm',
    'version' : '-'
}

def fetch_attack(attack, model, **config):
    dprint('Adversary', **config)

    if attack == 'AA':
        config['version'] = 'standard'
    elif attack == 'AA+':
        config['version'] = 'plus'

    if 'seed' in config and config['seed'] is None:
        config['seed'] = 0

    return ATTACK[attack](model, **config)
    
################################ GradAlign loss #######################################
# 通过对输入的真实标签与软标签对数概率log_prob的乘积求和，并取平均值来得到的。
#用onehot将输入标签转化为独热编码
# result它通过将原始标签（对应于独热编码中的1）乘以平滑因子，并将其余部分设置为 (factor - 1) / (10 - 1) 来实现。
# 这意味着对于不是原始标签的类别，我们将其概率稍微降低，而原始标签的概率稍微增加

def get_uniform_delta(shape, eps, requires_grad=True):
    delta = torch.zeros(shape).cuda()
    delta.uniform_(-eps, eps)
    delta.requires_grad = requires_grad
    return delta

def get_input_grad(model, X, y, eps, delta_init='none', backprop=False):
    if delta_init == 'none':
        delta = torch.zeros_like(X, requires_grad=True)
    # print( lower_limit.shape)
    # print( torch.sign(grad).shape)
    elif delta_init == 'random_uniform':
        delta = get_uniform_delta(X.shape, eps, requires_grad=True)
    elif delta_init == 'random_corner':
        delta = get_uniform_delta(X.shape, eps, requires_grad=True)
        delta = eps * torch.sign(delta)
    else:
        raise ValueError('wrong delta init')

    output = model(X + delta)
    loss = F.cross_entropy(output, y)
    grad = torch.autograd.grad(loss, delta, create_graph=True if backprop else False)[0]

    if not backprop:
        grad, delta = grad.detach(), delta.detach()
    return grad
#使用get_input_grad函数两次获取输入梯度，一次不进行反向传播（backprop=False），另一次进行随机均匀初始化并进行反向传播
'''使用余弦相似度（cosine_similarity）计算两个梯度之间的相似度。余弦相似度的范围是[-1, 1]，值越接近1表示两个向量越相似。
计算正则化项（reg），它是1 - cos.mean()与超参数args.gradalign_lambda的乘积。这个正则化项鼓励模型在不同条件下产生更相似的梯度
'''
def grad_align_loss(model, X, y, args):
    # grad1 = get_input_grad(model, X, y,  args.train_eps, delta_init='random_corner', backprop=False)
    grad1 = get_input_grad(model, X, y,  args.train_eps, delta_init='none', backprop=False)
    grad2 = get_input_grad(model, X, y,  args.train_eps, delta_init='random_uniform', backprop=True)
    grad1, grad2 = grad1.reshape(len(grad1), -1), grad2.reshape(len(grad2), -1)
    cos = torch.nn.functional.cosine_similarity(grad1, grad2, 1)
    reg = args.gradalign_lambda * (1.0 - cos.mean())

    return reg

################################ TRADES loss #######################################
def squared_l2_norm(x):
    flattened = x.view(x.unsqueeze(0).shape[0], -1)
    return (flattened ** 2).sum(1)

def l2_norm(x):
    return squared_l2_norm(x).sqrt()


def trades_loss(model,
                x_natural,
                y,
                optimizer,
                step_size=0.003,
                epsilon=0.031,
                perturb_steps=10,
                beta=1.0,
                distance='l_inf'):
    # define KL-loss
    criterion_kl = nn.KLDivLoss(size_average=False)
    model.eval()
    batch_size = len(x_natural)
    # generate adversarial example
    x_adv = x_natural.detach() + 0.001 * tc.randn(x_natural.shape).cuda().detach()
    
    if distance == 'l_inf':
        for _ in range(perturb_steps):
            x_adv.requires_grad_()
            with tc.enable_grad():
                loss_kl = criterion_kl(F.log_softmax(model(x_adv), dim=1),
                                       F.softmax(model(x_natural), dim=1))
            grad = tc.autograd.grad(loss_kl, [x_adv])[0]
            x_adv = x_adv.detach() + step_size * tc.sign(grad.detach())
            x_adv = tc.min(tc.max(x_adv, x_natural - epsilon), x_natural + epsilon)
            x_adv = tc.clamp(x_adv, 0.0, 1.0)
    elif distance == 'l_2':
        delta = 0.001 * tc.randn(x_natural.shape).cuda().detach()
        delta = Variable(delta.data, requires_grad=True)

        # Setup optimizers
        optimizer_delta = optim.SGD([delta], lr=epsilon / perturb_steps * 2)

        for _ in range(perturb_steps):
            adv = x_natural + delta

            # optimize
            optimizer_delta.zero_grad()
            with tc.enable_grad():
                loss = (-1) * criterion_kl(F.log_softmax(model(adv), dim=1),
                                           F.softmax(model(x_natural), dim=1))
            loss.backward()
            # renorming gradient
            grad_norms = delta.grad.view(batch_size, -1).norm(p=2, dim=1)
            delta.grad.div_(grad_norms.view(-1, 1, 1, 1))
            # avoid nan or inf if gradient is 0
            if (grad_norms == 0).any():
                delta.grad[grad_norms == 0] = tc.randn_like(delta.grad[grad_norms == 0])
            optimizer_delta.step()

            # projection
            delta.data.add_(x_natural)
            delta.data.clamp_(0, 1).sub_(x_natural)
            delta.data.renorm_(p=2, dim=0, maxnorm=epsilon)
        x_adv = Variable(x_natural + delta, requires_grad=False)
    else:
        x_adv = tc.clamp(x_adv, 0.0, 1.0)
    model.train()

    x_adv = Variable(tc.clamp(x_adv, 0.0, 1.0), requires_grad=False)
    # zero gradient
    optimizer.zero_grad()
    # calculate robust loss
    # logits = model(x_natural)
    # loss_natural = F.cross_entropy(logits, y)
    # loss_robust = (1.0 / batch_size) * criterion_kl(F.log_softmax(model(x_adv), dim=1),
    #                                                 F.softmax(model(x_natural), dim=1))
    # loss = loss_natural + beta * loss_robust
    # return model(x_adv), loss
    # calculate robust loss我的鲁棒损失更改，对抗样本和净化后的对抗样本更接近
    logits = model(x_natural)
    loss_natural = F.cross_entropy(logits, y)
    loss_robust = (1.0 / batch_size) * criterion_kl(F.log_softmax(model(x_adv), dim=1),
                                                    F.softmax(model(x_natural), dim=1))
    loss = loss_natural + beta * loss_robust
    return model(x_adv), loss


################################ Guided_Attack #######################################
def Guided_Attack(model,loss,image,target,eps=8/255,bounds=[0,1],steps=1,P_out=[],l2_reg=10,alt=1): 
    tar = Variable(target.cuda())
    img = image.cuda()
    eps = eps/steps 
    for step in range(steps):
        img = Variable(img,requires_grad=True)
        zero_gradients(img) 
        out  = model(img)
        R_out = nn.Softmax(dim=1)(out)
        cost = loss(out,tar) + alt*l2_reg*(((P_out - R_out)**2.0).sum(1)).mean(0) 
        cost.backward()
        per = eps * tc.sign(img.grad.data)
        adv = img.data + per.cuda() 
        img = tc.clamp(adv,bounds[0],bounds[1])
    return img