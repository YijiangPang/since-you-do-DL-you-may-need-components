from torchdiffeq import odeint_adjoint 
from torchdiffeq import odeint  
import torch.nn as nn
import torch
import numpy as np

      
class ODE_int_func(nn.Module):

    def __init__(self, odefunc, ODE_integration_time, flag_adjoint, tol):
        super(ODE_int_func, self).__init__()
        self.odefunc = odefunc
        self.integration_time = ODE_integration_time
        self.odeint = odeint if not flag_adjoint else odeint_adjoint
        self.tol = tol

    def forward(self, x):
        self.integration_time = self.integration_time.type_as(x)
        out = self.odeint(self.odefunc, x, self.integration_time, method = "adaptive_heun",rtol = self.tol, atol = self.tol)
        return out[1]

    @property
    def nfe(self):
        return self.odefunc.nfe

    @nfe.setter
    def nfe(self, value):
        self.odefunc.nfe = value


class ODE_base_func(nn.Module):
    feature_dim = 32
    def __init__(self, dim_in, T):
        super(ODE_base_func, self).__init__()
        self.T = T
        self.model_time = nn.Linear(1, self.feature_dim)
        self.model_feature = nn.Linear(dim_in, self.feature_dim)
        self.model_base = nn.Linear(self.feature_dim, dim_in)
        self.nfe = 0

    def forward(self, t, x):
        self.nfe += 1
        out = self.model_base(self.model_time(t.expand(x.shape[0], 1)) + self.model_feature(x))   #many other ways of imbedding time info into process
        return out


class NeuralODE_Model(nn.Module):
    ode_feature_dim = 32
    def __init__(self, ODE_integration_time, ODE_flag_adjoint, ODE_tol, dim_in, n_classes):
        super().__init__()
        self.in_block = nn.Sequential(nn.Linear(dim_in, self.ode_feature_dim))

        self.ODE_block = ODE_int_func(ODE_base_func(self.ode_feature_dim, ODE_integration_time[1]), ODE_integration_time, ODE_flag_adjoint, ODE_tol)

        self.fc = nn.Linear(self.ode_feature_dim, n_classes)

    def forward(self, x):
        out = self.in_block(x)
        out = self.ODE_block(out)
        out = self.fc(out)
        return out


if __name__ == "__main__":
    model = NeuralODE_Model(ODE_integration_time = torch.tensor([0, 10]), ODE_flag_adjoint = True, ODE_tol = 1e-4, dim_in = 1024, n_classes = 10)

    data_x = torch.tensor(np.random.random((100, 1024))).float()

    with torch.no_grad():
        yhat = model(data_x)
