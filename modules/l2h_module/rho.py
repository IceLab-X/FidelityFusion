import torch
from modules.l2h_module.base_l2h_module import Basic_l2h
from utils import smart_update
from utils.type_define import *

default_config = {
    'rho_value_init': 1.,
    'trainable': True,
}


class Res_rho_l2h(Basic_l2h):
    def __init__(self, config=None) -> None:
        super().__init__()
        self.config = smart_update(default_config, config)

        self.rho = torch.nn.Parameter(torch.tensor(self.config['rho_value_init'], dtype=torch.float32))
        if self.config['trainable']:
            self.rho.requires_grad = True
        else:
            self.rho.requires_grad = False

    # train
    # inputs = [x, y_low]       ->  [x]
    # outputs = [y_high]        ->  [y_res]
    def pre_process_at_train(self, inputs, outputs):
        x = inputs[0]
        y_low = inputs[1]

        y_high = outputs[0]

        re_present_inputs = [x[:y_high.shape[0]]]
        # re_present_outputs = [y_high - y_low*self.rho]

        re_present_outputs = [y_high - y_low[:y_high.shape[0]] * self.rho]
        return re_present_inputs, re_present_outputs

    def pre_process_at_predict(self, inputs, outputs):
        x = inputs[0]
        y_low = inputs[1]

        return [inputs[0]], outputs

    
    # predict
    # inputs = [x, y_low]       ->  [x, y_low]
    # outputs = [y_res]         ->  [y_high]
    def post_process_at_predict(self, inputs, outputs):
        x = inputs[0]
        y_low = inputs[1]

        res = outputs[0]

        # TODO: support res with var
        if isinstance(res, GP_val_with_var):
            res = res.get_mean()
        re_present_outputs = [y_low*self.rho + res]
        return inputs, re_present_outputs
    
    def get_train_params(self):
        return {'rho': self.rho}