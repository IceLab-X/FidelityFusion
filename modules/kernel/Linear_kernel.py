import torch
import os

class Linear_kernel(torch.nn.Module):
    # Linear kernel
    def __init__(self, exp_restrict, scale=1., bias=0.) -> None:
        super().__init__()
        self.exp_restrict = exp_restrict

        if exp_restrict is True:
            self.scale = torch.nn.Parameter(torch.log(torch.tensor(scale)))
        else:
            self.scale = torch.nn.Parameter(torch.tensor(scale))

        self.bias = torch.nn.Parameter(torch.tensor(bias))

    def forward(self, X, X2):
        # optimize for multi dim.
        if len(X.shape)>2 :
            assert len(X2.shape)>2, "X and X2 should be same dim"
            X = X.reshape(X.size(0), -1)
            X2 = X2.reshape(X2.size(0), -1)

        # TODO get X/X2 mean
        if self.exp_restrict is True:
            K = self.bias + self.scale.exp()*X @ X2.t()            
        else:
            K = self.bias + self.scale*X @ X2.t()
        return K

    def get_param(self, optimize_param_list):
        optimize_param_list.append(self.scale)
        optimize_param_list.append(self.bias)
        return optimize_param_list

    def set_param(self, param_list):
        with torch.no_grad():
            self.scale.copy_(param_list[0])
            self.bias.copy_(param_list[1])

    def get_params_need_check(self):
        _temp_list = []
        self.get_param(_temp_list)
        return _temp_list

    def clamp_to_positive(self):
        if 'GP_DEBUG' in os.environ and os.environ['GP_DEBUG'] == 'True':
            # if self.length_scale < 0:
                # print('DEBUG WARNING bias:{} clamp to 0'.format(self.bias.data))
                print('DEBUG WARNING scale:{} clamp to 0'.format(self.scale.data))
                print('\n')

        with torch.no_grad():
            # self.bias.copy_(self.bias.clamp_(min=0))
            self.scale.copy_(self.scale.clamp_(min=0))
