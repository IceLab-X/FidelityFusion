import os
import torch
import numpy as np
from utils.mlgp_log import mlgp_log

def nan_hook(self, inp, output):
    if not isinstance(output, tuple):
        outputs = [output]
    else:
        outputs = output

    for i, out in enumerate(outputs):
        nan_mask = torch.isnan(out)
        if nan_mask.any():
            mlgp_log.error("Nan value detect in ", self.__class__.__name__, "output")
            raise RuntimeError(f"NAN in output {i} at indices: ", nan_mask.nonzero(), "where:", out[nan_mask.nonzero()[:, 0].unique(sorted=True)])


def register_nan_hook(model):
    for submodule in model.modules():
        submodule.register_forward_hook(nan_hook)


class set_function_as_module_to_catch_error(torch.nn.Module):
    def __init__(self, custom_function, ) -> None:
        super().__init__()
        self.custom_function = custom_function

        # record to trace
        self.valid_args = None
        self.valid_kwargs = None

        # save path
        self.debug_path = './exception_log'

    def _find_availabla_path(self):
        if not os.path.exists(self.debug_path):
            os.mkdir(self.debug_path)
        index = 0
        while True:
            _path = os.path.join(self.debug_path, 'log_{}'.format(index))
            if not os.path.exists(_path):
                os.mkdir(_path)
                return _path
            else:
                index += 1

    def forward(self, *args, **kwargs):
        try:
            self.custom_function(*args, **kwargs)
            if self.valid_args is None:
                self.valid_args = args
                self.valid_kwargs = kwargs

        except Exception as e:
            _epoch = os.environ.get("mlgp_epoch", "Get Failed")
            _iter = os.environ.get("mlgp_iter", "Get Failed")
            mlgp_log.e("forward error on: {}".format(str(self.custom_function)))
            mlgp_log.e("reason: {}".format(e))
            mlgp_log.e("epoch: {}".format(_epoch))
            mlgp_log.e("iter: {}".format(_iter))

            log_path = self._find_availabla_path()
            jit_model = torch.jit.trace(self, self.valid_args, self.valid_kwargs)

            # save
            _np_args = [_v.cpu().detach().numpy() if hasattr(_v, 'numpy') else _v for _v in args]
            _np_kwargs = {}
            for _key, _v in kwargs.items():
                _np_kwargs[_key] = _v.cpu().detach().numpy() if hasattr(_v, 'numpy') else _v
            torch.jit.save(jit_model, os.path.join(log_path, 'module.pt'))
            np.save(os.path.join(log_path, 'args.npy'), _np_args)
            np.save(os.path.join(log_path, 'kwargs.npy'), _np_kwargs)

            with open(os.path.join(log_path, 'log.txt'), 'w') as f:
                f.write("forward error on: {}\n".format(str(self.custom_function)))
                f.write("reason: {}\n".format(e))
                f.write("epoch: {}\n".format(_epoch))
                f.write("iter: {}\n".format(_iter))
                f.flush()

            # following code only work on mlgp repository, optimize the following code if you use this file on others repository.
            mlgp_record_file = os.environ.get('mlgp_record_file', None)
            if mlgp_record_file is not None:
                with open(mlgp_record_file, 'a') as f:
                    f.write('@record_result@\n')
                    f.write('@exception_catch_log@\n')
                    f.write("forward error on: {}\n".format(str(self.custom_function)))
                    f.write("reason: {}\n".format(e))
                    f.write("epoch: {}\n".format(_epoch))
                    f.write("iter: {}\n".format(_iter))
                    f.write("log file path: {}\n".format(log_path))
                    f.write('@exception_catch_log@\n')
                    f.flush()

        return self.custom_function(*args, **kwargs)
    
    