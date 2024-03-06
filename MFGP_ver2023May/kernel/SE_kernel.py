import torch


class SE_kernel(torch.nn.Module):
    # Squared Exponential Kernel
    def __init__(self, noise_exp_format, length_scale=1., scale=1.) -> None:
        super().__init__()
        self.noise_exp_format = noise_exp_format

        length_scale = torch.tensor(length_scale)
        scale = torch.tensor(scale)

        if noise_exp_format is True:
            self.length_scale = torch.nn.Parameter(torch.log(length_scale))
            self.scale = torch.nn.Parameter(torch.log(scale))
        else:
            self.length_scale = torch.nn.Parameter(length_scale)
            self.scale = torch.nn.Parameter(scale)

    def forward(self, X, X2):
        if self.noise_exp_format is True:
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
        K = scale * torch.exp(-0.5 * K)

        return K


if __name__ == '__main__':
    print('test1')
    ke = SE_kernel(True)

    print('test2')
    ke = SE_kernel(False, 2.)

    print('test3')
    ke = SE_kernel(False, 2., 3.)