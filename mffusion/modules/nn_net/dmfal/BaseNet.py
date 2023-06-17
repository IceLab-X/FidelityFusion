import numpy as np
import torch
import torch.distributions as distributions
torch.set_default_tensor_type(torch.DoubleTensor)


class AdaptiveBaseNet:
    
    def __init__(self, layers, activation, device, torch_type):
        
        self.device = device
        self.torch_type = torch_type
        
        self.layers = layers
        self.num_layers = len(self.layers)
        self.activation = {'tanh': torch.nn.Tanh(), 'relu': torch.nn.ReLU(), 'sigmoid': torch.nn.Sigmoid()}[activation]
        
        self.input_dim = self.layers[0]
        self.latent_dim = self.layers[-3]
        self.base_dim = self.layers[-2]
        self.output_dim = self.layers[-1]
        
        self.latent_weights = []
        self.latent_biases = []
        
        for l in range(self.num_layers-3):
            W = self._xavier_init(self.layers[l], self.layers[l+1])
            b = torch.zeros([1,self.layers[l+1]], dtype=self.torch_type, device=self.device, requires_grad=True)
            self.latent_weights.append(W)
            self.latent_biases.append(b)
        
        self.W_mu = self._xavier_init(self.latent_dim, self.base_dim)
        self.b_mu = torch.zeros([1,self.base_dim], dtype=self.torch_type, device=self.device, requires_grad=True)
        
        self.W_rho = torch.zeros([self.latent_dim, self.base_dim], dtype=self.torch_type, device=self.device, requires_grad=True)
        self.W_std = torch.log(1 + torch.exp(self.W_rho))
        
        self.b_rho = torch.zeros([1, self.base_dim], dtype=self.torch_type, device=self.device, requires_grad=True)
        self.b_std = torch.log(1 + torch.exp(self.b_rho))
        
        self.A = self._xavier_init(self.base_dim, self.output_dim)
        self.A_b = torch.zeros([1,self.output_dim], dtype=self.torch_type, device=self.device, requires_grad=True)
        
        self.normal = distributions.normal.Normal(
            torch.tensor([0.0], dtype=self.torch_type, device=self.device), 
            torch.tensor([1.0], dtype=self.torch_type, device=self.device)
        )


        self.kld = self._eval_kld()
        self.reg = self._eval_reg()
        
    
    def _sample_from_posterior(self,):
        epsi_W = torch.squeeze(self.normal.sample(self.W_mu.shape))
        W_sample = self.W_mu + self.W_std*epsi_W
        
        epsi_b = torch.squeeze(self.normal.sample(self.b_mu.shape), dim=2)
        b_sample = self.b_mu + self.b_std*epsi_b
        
        return W_sample, b_sample
    
    def forward(self, X, sample=False):
        
        H = X.double()
        for l in range(self.num_layers-3):
            W = self.latent_weights[l].double()
            b = self.latent_biases[l].double()
            H = torch.add(torch.matmul(H, W), b)
            
            # scale before the nonlinear-op
            in_d = self.layers[l]
            H = H/np.sqrt(in_d+1)
            H = self.activation(H) 
        
        # project the latent base to base
        if sample:
            W_sample, b_sample = self._sample_from_posterior()
            W_sample = W_sample.double()
            b_sample = b_sample.double()
            H = torch.add(torch.matmul(H, W_sample), b_sample)
        else:
            self.W_mu = self.W_mu.double()
            self.b_mu = self.b_mu.double()
            H = torch.add(torch.matmul(H, self.W_mu), self.b_mu)
        #
        
        base = H/np.sqrt(self.latent_dim+1)
        self.A = self.A.double()
        self.A_b = self.A_b.double()
        Y = torch.add(torch.matmul(base, self.A), self.A_b)
        Y = Y/np.sqrt(self.base_dim+1)

        return Y, base
    
    def forward_base_by_sample(self, X, W_sample, b_sample):
        
        H = X
        for l in range(self.num_layers-3):
            W = self.latent_weights[l]
            b = self.latent_biases[l]
            H = torch.add(torch.matmul(H, W), b)
            
            # scale before the nonlinear-op
            in_d = self.layers[l]
            H = H/np.sqrt(in_d+1)
            H = self.activation(H) 
        #
        
        H = torch.add(torch.matmul(H, W_sample), b_sample)
        
        base = H/np.sqrt(self.latent_dim+1)

        return base
        
        
    def _eval_reg(self,):
        L2_norm_list = []
        for w in self.latent_weights:
            L2_norm_list.append(torch.sum(torch.square(w)))
        #
        for b in self.latent_biases:
            L2_norm_list.append(torch.sum(torch.square(b)))
        #
        L2_norm_list.append(torch.sum(torch.square(self.A)))
        L2_norm_list.append(torch.sum(torch.square(self.A_b)))
        
        return sum(L2_norm_list)
        
    
    def _eval_kld(self,):
        kld_W = torch.sum(-torch.log(self.W_std) + 0.5*(torch.square(self.W_std) + torch.square(self.W_mu)) - 0.5)
        kld_b = torch.sum(-torch.log(self.b_std) + 0.5*(torch.square(self.b_std) + torch.square(self.b_mu)) - 0.5)
        
        return kld_W+kld_b
    
    def _xavier_init(self, in_dim, out_dim):
        xavier_stddev = np.sqrt(2.0/(in_dim + out_dim))
        W = torch.normal(size=(in_dim, out_dim), mean=0.0, std=xavier_stddev, requires_grad=True, device=self.device, dtype=self.torch_type)
        return W
    
    def _msra_init(self, in_dim, out_dim):
        xavier_stddev = np.sqrt(2.0/(in_dim))
        W = torch.normal(size=(in_dim, out_dim), mean=0.0, std=xavier_stddev, requires_grad=True, device=self.device, dtype=self.torch_type)
        return W
    
    def parameters(self,):
        params = {}
        params['latent_weights'] = self.latent_weights
        params['latent_biases'] = self.latent_biases
        params['W_mu'] = self.W_mu
        params['W_rho'] = self.W_rho
        params['b_mu'] = self.b_mu
        params['b_rho'] = self.b_rho
        params['A'] = self.A
        params['A_b'] = self.A_b
        
        return params
        

class DropoutBaseNet:
    
    def __init__(self, layers, activation, dropout=0.2):
        
        self.layers = layers
        self.num_layers = len(self.layers)
        self.activation = {'tanh': torch.nn.Tanh(), 'relu': torch.nn.ReLU(), 'sigmoid': torch.nn.Sigmoid()}[activation]
        self.dropout = dropout
        
        self.input_dim = self.layers[0]
        self.output_dim = self.layers[-1]
        self.latent_dim = self.layers[-3]
        self.base_dim = self.layers[-2]
        
        self.base_weights = []
        self.base_biases = []
        
        for l in range(self.num_layers-2):
            W = self._xavier_init(self.layers[l], self.layers[l+1])
            b = torch.zeros([1,self.layers[l+1]], dtype=self.torch_type, device=self.device, requires_grad=True)
            self.base_weights.append(W)
            self.base_biases.append(b)
        
        
        self.A = self._xavier_init(self.base_dim, self.output_dim)
        self.A_b = torch.zeros([1,self.output_dim], dtype=self.torch_type, device=self.device, requires_grad=True)

        self.reg = self._eval_reg()
        
    
    def forward(self, X, sample=False):
        
        H = torch.nn.functional.dropout(X, p=self.dropout, training=sample)
        for l in range(self.num_layers-2):
            W = self.base_weights[l]
            b = self.base_biases[l]
            H = torch.add(torch.matmul(H, W), b)
            
            # scale before the nonlinear-op
            in_d = self.layers[l]
            H = H/np.sqrt(in_d+1)
            H = self.activation(H)
            H = torch.nn.functional.dropout(H, p=self.dropout, training=sample)
        
        base = H/np.sqrt(self.latent_dim+1)
        
        Y = torch.add(torch.matmul(base, self.A), self.A_b)
        Y = Y/np.sqrt(self.base_dim+1)

        return Y, base
        
        
    def _eval_reg(self,):
        L2_norm_list = []
        for w in self.base_weights:
            # print(w.shape)
            L2_norm_list.append(torch.sum(torch.square(w)))
        #

        L2_norm_list.append(torch.sum(torch.square(self.A)))
        
        return sum(L2_norm_list)

    def _xavier_init(self, in_dim, out_dim):
        xavier_stddev = np.sqrt(2.0/(in_dim + out_dim))
        W = torch.normal(size=(in_dim, out_dim), mean=0.0, std=xavier_stddev, requires_grad=True, device=self.device, dtype=self.torch_type)
        return W
    
    def _msra_init(self, in_dim, out_dim):
        xavier_stddev = np.sqrt(2.0/(in_dim))
        W = torch.normal(size=(in_dim, out_dim), mean=0.0, std=xavier_stddev, requires_grad=True, device=self.device, dtype=self.torch_type)
        return W
    
    def parameters(self,):
        params = {}
        params['base_weights'] = self.base_weights
        params['base_biases'] = self.base_biases
        params['A'] = self.A
        params['A_b'] = self.A_b
        
        return params
        

