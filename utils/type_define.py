
class GP_val_with_var:
    def __init__(self, mean, var) -> None:
        self.mean = mean
        self.var = var

    def get_mean(self):
        return self.mean
    
    def get_var(self):
        return self.var
    
    def reg_func(self, func, *args, **kwargs):
        self.mean = func(self.mean, *args, **kwargs)
        self.var = func(self.var, *args, **kwargs)