import torch
from utils.type_define import GP_val_with_var

'''
# DOC: 动态注册类的操作, 有利于兼容性, 但是不利于代码可读性, 对初学者不友好, 目前不启用
def _register_func(self, class_operation, attr, attr_operation):
    def func(self, *args, **kwargs):
        return getattr(getattr(self, attr), attr_operation)(*args, **kwargs)
    _func = func
    setattr(self, class_operation, _func)

def register_standard_operation(self):
    if self.dnm is not None:
        self._register_func('normalize', 'dnm', 'normalize_all')
        self._register_func('denormalize', 'dnm', 'denormalize_outputs')

    if self.gp_model is not None:
        self._register_func('gp_train', 'gp_model', 'train')
        self._register_func('gp_predict', 'gp_model', 'predict')

    if self.pre_process_block is not None:
        self._register_func('pre_process', 'pre_process_block', 'pre_process')
        self._register_func('post_process', 'post_process_block', 'post_process')
    return
'''

class GP_model_block(torch.nn.Module):
    # ======= forward step =======
    #   1.normalize
    #   2.preprocess
    #   3.model predict
    #   4.postprocess
    #   5.denormalize

    # ======= backward step =======
    #   1.normalize
    #   2.preprocess
    #   3.model compute loss

    def __init__(self) -> None:
        super().__init__()
        self.dnm = None
        self.gp_model = None
        self.pre_process_block = None
        self.post_process_block = None

    # @tensor_wrap_to_tensor_with_uncertenty
    def predict(self, inputs):
        inputs = self.dnm.normalize_inputs(inputs)

        if self.pre_process_block is not None:
            gp_inputs, _ = self.pre_process_block.pre_process_at_predict(inputs, None)
        else:
            gp_inputs = inputs

        gp_outputs = self.gp_model.predict(gp_inputs)

        if self.post_process_block is not None:
            _, outputs = self.post_process_block.post_process_at_predict(inputs, gp_outputs)
        else:
            outputs = gp_outputs

        outputs = self.dnm.denormalize_outputs(outputs)
        return outputs

    def predict_with_detecing_subset(self, inputs):
        inputs = self.dnm.normalize_inputs(inputs)

        if self.pre_process_block is not None:
            gp_inputs, _ = self.pre_process_block.pre_process_at_predict(inputs, None)
        else:
            gp_inputs = inputs

        from utils.subset_tools import Subset_check
        src_inputs = self.gp_model.inputs_tr
        checker = Subset_check(src_inputs[0])

        _, subset_index = checker.get_subset(gp_inputs[0])
        _, nonsubset_index = checker.get_non_subset(gp_inputs[0])

        subset_outputs = [_v[subset_index] for _v in self.gp_model.outputs_tr]
        if subset_outputs[0].shape[0] == 0:
            subset_outputs = None

        nonsubset_inputs = [_v[nonsubset_index] for _v in gp_inputs]
        if nonsubset_inputs[0].shape[0]!= 0:
            nonsubset_outputs = self.gp_model.predict(nonsubset_inputs)
        else:
            nonsubset_outputs = None

        if subset_outputs is None:
            outputs = nonsubset_outputs
        elif nonsubset_outputs is None:
            outputs = []
            for _v in subset_outputs:
                outputs.append(GP_val_with_var(_v, torch.zeros_like(_v)))
        else:
            outputs = []
            if isinstance(nonsubset_outputs[0], GP_val_with_var):
                gp_output_mean = [_v.mean for _v in nonsubset_outputs]
                gp_output_var = [_v.var for _v in nonsubset_outputs]
            else:
                gp_output_mean = [_v for _v in nonsubset_outputs]
                gp_output_var = [torch.zeros_like(_v) for _v in nonsubset_outputs]
            for i in range(len(nonsubset_outputs)):
                mix_mean = torch.cat([subset_outputs[i], gp_output_mean[i]], axis=0)
                mix_var = torch.cat([torch.zeros_like(subset_outputs[i]), gp_output_var[i]], axis=0)
                outputs.append(GP_val_with_var(mix_mean, mix_var))

        outputs = self.dnm.denormalize_outputs(outputs)
        return outputs

    # 当前这种形式会带来额外的内存,计算开销. 优点是自由度高,容易添加新的操作
    def compute_loss(self, inputs, outputs):
        inputs, outputs = self.dnm.normalize_all(inputs, outputs)

        if self.pre_process_block is not None:
            inputs, outputs = self.pre_process_block.pre_process_at_train(inputs, outputs)
        
        loss = self.gp_model.compute_loss(inputs, outputs)
        return loss

    def save_model(self, path):
        # TODO
        pass

    def load_model(self, path):
        # TODO
        pass

    def sync_block_after_train(self):
        # TODO
        pass

    def get_train_params(self):
        params_dict = {}
        params_dict.update(self.gp_model.get_train_params())
        if self.pre_process_block is not None:
            params_dict.update(self.pre_process_block.get_train_params())
        if self.post_process_block is not None:
            params_dict.update(self.post_process_block.get_train_params())
        return params_dict



if __name__ == '__main__':

    pass