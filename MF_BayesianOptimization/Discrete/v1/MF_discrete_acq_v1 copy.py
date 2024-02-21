import torch
import torch.nn as nn

class DiscreteAcquisitionFunction(nn.Module):
    def __init__(self, ):
        super().__init__()
        self.params = {}

    def set_params(self, **kwargs):
        self.params.update(kwargs)

    def get_params(self):
        return self.params

    def evaluate(self, x):
        raise NotImplementedError("Subclasses must implement the evaluate method")
    
    def MF_acq_optimise_x(self, niteration, lr):
        optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        # optimizer.zero_grad()
        for i in range(niteration):
            optimizer.zero_grad()
            loss = self.negative_ucb(fidelity_indicator)
            loss.backward()
            optimizer.step()
            self.x.data.clamp_(0.0, 1.0)
            print('iter', i, 'x:', self.x, 'loss_negative_ucb:',loss.item(), end='\n')

    
    def MF_acq_next_x():
        return 0
        
    def MF_acq_next_s(mf_acq_func, search_range, model_cost, seed):
        return 0
        


class UCB(BaseAcquisitionFunction):
    def __init__(self, ):
        super(UCB, self).__init__()
        # self.exploration_parameter = exploration_parameter
        # self.x = nn.Parameter(torch.ones(x_dimension))

    def evaluate(self, mean_func, var_func, cost_func):
        # UCB-specific logic using self.exploration_parameter
        # ...
        self.x = nn.Parameter(torch.ones(x_dimension))

        return x


# class ES(BaseAcquisitionFunction):
#     def __init__(self, temperature):
#         super(ES, self).__init__()
#         self.temperature = temperature

#     def evaluate(self, x):
#         # ES-specific logic using self.temperature
#         # ...
#         return x


# class KG(BaseAcquisitionFunction):
#     def __init__(self, confidence_level):
#         super(KG, self).__init__()
#         self.confidence_level = confidence_level

#     def evaluate(self, x):
#         # KG-specific logic using self.confidence_level
#         # ...
#         return x

# Example usage:
ucb_acquisition = UCB(exploration_parameter=1.0)
ucb_acquisition.set_params(foo=42)

# es_acquisition = ES(temperature=0.1)
# es_acquisition.set_params(bar=3.14)

# kg_acquisition = KG(confidence_level=0.95)
# kg_acquisition.set_params(baz="hello")
