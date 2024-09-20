import torch
import torch.nn as nn

class upper_confidence_bound_continuous(nn.Module):
    def __init__(self, x_dimension, search_range, posterior_function, model_cost, data_manager,norm, seed):
        super(upper_confidence_bound_continuous, self).__init__()
        torch.manual_seed(seed[0])
        self.z_range = torch.tensor(sorted(torch.rand(100) * (search_range[-1][1] - search_range[-1][0]) + search_range[-1][0])).reshape(-1, 1)

        # 传入的用于计算的函数/参数
        self.search_range = search_range
        self.pre_func = posterior_function
        self.model_cost = model_cost
        self.x_dimension = x_dimension
        self.data_manager = data_manager
        self.log_length_scale = nn.Parameter(torch.zeros(x_dimension))    # ARD length scale
        self.log_scale = nn.Parameter(torch.zeros(1))   # kernel scale
        self.x_norm = norm[0]
        self.y_norm = norm[1]
        # select criteria
        self.seed = seed[0]
        self.beta = torch.tensor(1.0)
        self.d = x_dimension
        self.k_0 = torch.tensor(1.0)
        self.p = torch.tensor(1.0)

    def kernel(self, X1, X2):
    # the common RBF kernel
        X1 = X1 / self.log_length_scale.exp()
        X2 = X2 / self.log_length_scale.exp()
        X1_norm2 = torch.sum(X1 * X1, dim=1).view(-1, 1)
        X2_norm2 = torch.sum(X2 * X2, dim=1).view(-1, 1)

        K = -2.0 * X1 @ X2.t() + X1_norm2.expand(X1.size(0), X2.size(0)) + X2_norm2.t().expand(X1.size(0), X2.size(0))  #this is the effective Euclidean distance matrix between X1 and X2.
        K = self.log_scale.exp() * torch.exp(-0.5 * K)
        return K

    def information_gap(self, input):
        if input == None:
            input = self.z_range
        else:
            input = torch.ones(1).reshape(-1, 1)*input

        phi = self.kernel(input.double(), torch.ones(1).reshape(-1, 1).double())
        phi = phi.detach()
        ksin = torch.sqrt(1- torch.pow(phi, 2))
        return ksin

    def gamma_z(self, ksin_z):
        q = 1 / (self.p + self.d + 2)
        lambda_balance = torch.pow(self.model_cost.compute_cost(self.z_range)/self.model_cost.compute_cost(1), q)
        gamma_z = torch.sqrt(self.k_0) * ksin_z * lambda_balance
        return gamma_z

    def negative_ucb(self):
        # mean, var = self.pre_func(self.x, torch.ones(1).reshape(-1, 1)*self.search_range[-1][-1])
        x_in = torch.cat((self.x, torch.ones(1).reshape(-1, 1)*self.search_range[-1][-1]), dim=1)
        x_in = self.x_norm.normalize(x_in)
        mean, var = self.pre_func(self.data_manager, x_in)
        mean = self.y_norm.denormalize(mean)
        ucb = mean[:,0] + self.beta * var
        return -ucb

    def optimise_adam(self, niteration, lr):
        optimizer = torch.optim.Adam(self.parameters(), lr=lr)

        for i in range(niteration):
            optimizer.zero_grad()
            loss = self.negative_ucb()
            loss.backward()
            optimizer.step()
            print('iter'+str(i)+'/'+str(niteration), 'loss_negative_ucb:', loss.item(), end='\r')

    def compute_next(self):

        # optimize x
        torch.manual_seed(self.seed + 10086)
        tem = []
        for i in range(self.x_dimension):
            tt = torch.rand(1, 1) * (self.search_range[i][1] - self.search_range[i][0]) + self.search_range[i][0]
            tem.append(tt)
        tt = torch.cat(tem, dim=1)
        print(tt)
        self.x = nn.Parameter(tt.reshape(1, self.x_dimension).double(),  requires_grad=True)
        self.optimise_adam(niteration=10, lr=0.01)

        new_x = self.x.detach()

        tau_z_mean = []
        tau_z_std = []

        tau_z_mean = []
        tau_z_std = []
        for z in self.z_range:
            z = z.reshape(-1, 1)
            x_te = torch.cat((new_x, z), dim=1)
            x_te = self.x_norm.normalize(x_te)
            m, v = self.pre_func(self.data_manager, x_te)
            m = self.y_norm.denormalize(m)
            tau_z_mean.append(m.detach().numpy())
            tau_z_std.append(torch.sqrt(v.detach()))

        ksin_z = self.information_gap(None)
        gamma_z = self.gamma_z(ksin_z)

        possible_z = []
        for i in range(self.z_range.shape[0]):
            condition_1 = tau_z_std[i][0][0] > gamma_z[i]
            condition_2 = ksin_z[i] > self.information_gap(torch.sqrt(self.p)) / torch.sqrt(self.beta)
            if condition_1 and condition_2:
                possible_z.append(self.z_range[i])

        if len(possible_z) == 0:
            new_s = 0.1
        else:
            new_s = min(possible_z)

        # if isinstance(new_x, torch.Tensor):
        #     new_x = new_x.detach().numpy()

        return new_x, new_s
