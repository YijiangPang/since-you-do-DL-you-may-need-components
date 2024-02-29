from ZerothOptimizer.OptimizerBase import OptimizerBase


class MeZO(OptimizerBase):
    def __init__(self, **kwargs):
        OptimizerBase.__init__(self, **kwargs)

    def zo_step(self, model, loss_func, x, y):
        self.init_zo_randomness(x)
        # First function evaluation
        self.zo_perturb_parameters(scaling_factor=1)
        loss1 = self.zo_forward(model, loss_func, x, y)

        # Second function evaluation
        self.zo_perturb_parameters(scaling_factor=-2)
        loss2 = self.zo_forward(model, loss_func, x, y)

        self.projected_grad = (loss1 - loss2) / (2 * self.eps)

        # Reset model back to its parameters at start of step
        self.zo_perturb_parameters(scaling_factor=1)

        self.zo_update()
        loss_step = (loss1 + loss2)/2
        self._zo_update(loss_step, self.projected_grad)   #update LR
        return loss_step
    
    def zo_update(self):
        self.zo_random_gen.set_state(self.zo_random_gen_initial_state)
        for name, param in self.named_parameters_to_optim:
            # Resample z
            z = self.noise_sample(param, self.zo_random_gen)
            if "bias" not in name and "layer_norm" not in name and "layernorm" not in name:
                param.data = param.data - self.lr * (self.projected_grad * z + self.weight_decay_rate * param.data)
            else:
                param.data = param.data - self.lr * (self.projected_grad * z)
    
    def zo_perturb_parameters(self, scaling_factor=1):
        self.zo_random_gen.set_state(self.zo_random_gen_initial_state)
        for name, param in self.named_parameters_to_optim:
            z = self.noise_sample(param, self.zo_random_gen)
            param.data = param.data + scaling_factor * z * self.eps

    def _zo_update(self, loss, grad_norm):
        self.lr = self.lr_cosine_decay(self.lr_init, self.T_current, self.T_max)
        self.T_current = min(self.T_current + 1, self.T_max)
        self.loss_record.append(loss) 
        self.lr_record.append(abs(self.lr*grad_norm))   #true LR
        if grad_norm is not None: self.grad_app_record.append(grad_norm)