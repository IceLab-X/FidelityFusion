import torch

default_config = {
}

class Basic_l2h(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def pre_process_at_train(self, inputs, outputs):
        pass

    # normally needn't
    def pre_process_at_predict(self, inputs, outputs):
        pass

    # normally needn't
    def post_process_at_train(self):
        pass

    def post_process_at_predict(self):
        pass

    def get_train_params(self):
        pass