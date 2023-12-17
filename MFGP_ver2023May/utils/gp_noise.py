import torch

default_config = {
    'format': 'exp',
    'init_value': 1e-3
}


class GP_noise_box(torch.nn.Module):
    def __init__(self, noise_config):
        super().__init__()
        assert noise_config['format'] in ['exp', 'linear'], "noise format should be 'exp' or 'linear'"
        self.config = noise_config
        self.format = noise_config['format']

        if self.format == 'exp':
            self.value = torch.nn.Parameter(torch.log(torch.tensor(noise_config['init_value'], dtype=torch.float32)))
        else:
            self.value = torch.nn.Parameter(torch.tensor(noise_config['init_value'], dtype=torch.float32))

    def get(self):
        if self.format == 'exp':
            return torch.exp(self.value)
        else:
            return self.value