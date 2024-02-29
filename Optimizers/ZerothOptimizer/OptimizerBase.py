import torch
import math
import numpy as np

class OptimizerBase:
    loss_func_default = torch.nn.functional.cross_entropy
    def __init__(self, model, lr, eps_init, eps_end, T_max, weight_decay_rate):
        self.named_parameters_to_optim = []
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.named_parameters_to_optim.append((name, param))
        self.model = model
        self.lr_init, self.lr = lr, lr
        self.eps_init, self.eps, self.eps_end = eps_init, eps_init, eps_end
        self.T_max = T_max
        self.weight_decay_rate = weight_decay_rate
        self.T_current = 0
        self.loss_record, self.lr_record, self.grad_app_record = [], [], []
    
    def zo_forward(self, model, loss_func, x, y):
        model.eval()
        loss_func = self.loss_func_default if loss_func is None else loss_func
        with torch.inference_mode():
            if y is not None:
                loss = loss_func(model(x), y) 
            else:
                outputs = model(**x)
                loss = outputs["loss"] if isinstance(outputs, dict) else outputs[0]
                if torch.cuda.device_count() > 1:
                    loss = loss.mean()  # mean() to average on multi-gpu parallel training
        return loss.data.detach().cpu().item()
    
    #need to be called by "zo_update" of child class
    def _zo_update(self, loss, grad_norm):
        pass    #update model paras
        self.lr = self.lr_cosine_decay(self.lr_init, self.T_current, self.T_max)
        self.T_current = min(self.T_current + 1, self.T_max)
        self.loss_record.append(loss) 
        self.lr_record.append(self.lr)
        if grad_norm is not None: self.grad_app_record.append(grad_norm)

    def init_zo_randomness(self, x):
        zo_random_seed = np.random.randint(1000000000)
        device = x.device if not isinstance(x, dict) else x["labels"].device
        self.zo_random_gen = torch.Generator(device).manual_seed(zo_random_seed)
        self.zo_random_gen_initial_state = self.zo_random_gen.get_state()

    def noise_sample(self, param, zo_random_gen):
        z = torch.zeros_like(param).to(param).normal_(mean=0, std=1, generator = zo_random_gen)
        return z

    def step(self, model, loss_func, x, y):
        self.zo_step(model, loss_func, x, y)
    
    @staticmethod
    def lr_cosine_decay(initial_lr, current_step, total_steps, min_lr = 0.0):
        if current_step >= total_steps:
            return min_lr
        angle = math.pi * current_step / total_steps
        decayed_lr = (initial_lr - min_lr) * 0.5 * (1 + math.cos(angle)) + min_lr
        return decayed_lr