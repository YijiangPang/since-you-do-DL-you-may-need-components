import torch
import torch.nn.functional as F
import math
import scipy.optimize as sopt

#Hashimoto, Tatsunori, et al. "Fairness without demographics in repeated loss minimization." International Conference on Machine Learning. PMLR, 2018.

def chi2_loss(logits, y, alpha = 0.05):
    max_l = 10.
    C = math.sqrt(1 + (1 / alpha - 1) ** 2)

    loss_task = F.cross_entropy(logits, y, reduction = "none")

    foo = lambda eta: C * math.sqrt((F.relu(loss_task - eta) ** 2).mean().item()) + eta
    opt_eta = sopt.brent(foo, brack=(0, max_l))
    loss_task = C * torch.sqrt((F.relu(loss_task - opt_eta) ** 2).mean()) + opt_eta

    return loss_task


def cvar_loss(logits, y, alpha = 0.05):
    batch_size = logits.shape[0]
    n1 = int(alpha * batch_size)
    loss = F.cross_entropy(logits, y, reduction = "none")
    rk = torch.argsort(loss, descending=True)
    loss = loss[rk[:n1]].mean()
    return loss

def irm_loss(model, logits, y, l2_penalty_weight = 0.001, grad_penalty_weight = 10000):
    loss_task = F.cross_entropy(logits, y)
    #grad penalty
    def grad_penalty_func(logits, y):
        scale = torch.tensor(1.).to(logits).requires_grad_()
        loss = F.cross_entropy(logits * scale, y)
        grad = torch.autograd.grad(loss, [scale], create_graph=True)[0]
        return torch.sum(grad ** 2)
    loss_grad_penalty = grad_penalty_func(logits, y)
    #l2 penalty
    weight_norm = torch.tensor(0.).to(logits)
    for w in model.parameters():
        weight_norm += w.norm().pow(2)
    
    loss = loss_task + l2_penalty_weight * weight_norm + grad_penalty_weight * loss_grad_penalty

    return loss


def ib_irm_loss(model, logits, y, feature, num_classes = 2, l2_penalty_weight = 0.001, grad_penalty_weight = 10000, ib_penalty_weight = 1e1):
    loss_task = F.cross_entropy(logits, y)
    #grad penalty
    def grad_penalty_func(logits, y):
        scale = torch.tensor(1.).to(y.device).requires_grad_()
        loss = F.cross_entropy(logits * scale, y)
        grad = torch.autograd.grad(loss, [scale], create_graph=True)[0]
        return torch.sum(grad ** 2)
    loss_grad_penalty = grad_penalty_func(logits, y)
    #l2 penalty
    weight_norm = torch.tensor(0.).to(logits)
    for w in model.parameters():
        weight_norm += w.norm().pow(2)
    #feature var penalty
    index_class = [y == i for i in range(num_classes)]
    loss_var_penalty = [torch.mean(torch.var(feature[ind], dim = 1)) for ind in index_class]
    loss_var_penalty = [v for v in loss_var_penalty if not torch.isnan(v)]
    loss_var_penalty = sum(loss_var_penalty)/sum(1 for _ in loss_var_penalty) if sum(1 for _ in loss_var_penalty) > 0 else 0
    # loss_var_penalty = torch.mean(torch.var(feature, dim = 1))
    
    loss = loss_task + l2_penalty_weight * weight_norm + grad_penalty_weight * loss_grad_penalty + ib_penalty_weight*loss_var_penalty

    return loss