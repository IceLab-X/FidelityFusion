import torch
import os

class RQ_kernel(torch.nn.Module):
    # Rational Quadratic Kernel
    def __init__(self, exp_restrict = 'exp', length_scale=1., scale=1., alpha=2.) -> None:
        super().__init__()
        self.exp_restrict = exp_restrict

        if exp_restrict is True:
            self.length_scale = torch.nn.Parameter(torch.log(torch.tensor(length_scale)))
            self.scale = torch.nn.Parameter(torch.log(torch.tensor(scale)))
        else:
            self.length_scale = torch.nn.Parameter(torch.tensor(length_scale))
            self.scale = torch.nn.Parameter(torch.tensor(scale))

        self.alpha = torch.nn.Parameter(torch.tensor(alpha))
        # TODO should we add noise here?

    def forward(self, X, X2):
        if self.exp_restrict is True:
            length_scale = torch.exp(self.length_scale).view(1, -1)
            scale = torch.exp(self.scale).view(1, -1)
        else:
            length_scale = self.length_scale.view(1, -1)
            scale = self.scale.view(1, -1)

        # optimize for multi dim.
        if len(X.shape)>2 :
            assert len(X2.shape)>2, "X and X2 should be same dim"
            X = X.reshape(X.size(0), -1)
            X2 = X2.reshape(X2.size(0), -1)

        X = X / length_scale.expand(X.size(0), length_scale.size(1))
        X2 = X2 / length_scale.expand(X2.size(0), length_scale.size(1))

        X_norm2 = torch.sum(X * X, dim=1).view(-1, 1)
        X2_norm2 = torch.sum(X2 * X2, dim=1).view(-1, 1)

        # compute effective distance
        K = -2.0 * X @ X2.t() + X_norm2.expand(X.size(0), X2.size(0)) + X2_norm2.t().expand(X.size(0), X2.size(0))
        K = scale * torch.pow(1 + K/(2*self.alpha), -self.alpha)

        return K

    # def get_param(self):
    #     return self.length_scale, self.log_scale

    def get_param(self, optimize_param_list):
        optimize_param_list.append(self.length_scale)
        optimize_param_list.append(self.scale)
        optimize_param_list.append(self.alpha)
        return optimize_param_list

    def set_param(self, param_list):
        with torch.no_grad():
            self.length_scale.copy_(param_list[0])
            self.scale.copy_(param_list[1])
            self.alpha.copy_(param_list[2])

    def get_params_need_check(self):
        _temp_list = []
        self.get_param(_temp_list)
        return _temp_list

    def clamp_to_positive(self):
        if 'GP_DEBUG' in os.environ and os.environ['GP_DEBUG'] == 'True':
            # if self.length_scale < 0:
                print('DEBUG WARNING length_scale:{} clamp to 0'.format(self.length_scale.data))
                print('DEBUG WARNING scale:{} clamp to 0'.format(self.scale.data))
                print('\n')

        with torch.no_grad():
            self.length_scale.copy_(self.length_scale.clamp_(min=0))
            self.scale.copy_(self.scale.clamp_(min=0))
