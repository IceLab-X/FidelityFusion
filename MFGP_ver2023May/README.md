# File list
```
├─gp
│  ├─base_gp
│  │  │  cigp.py
│  │  │  fides.py
│  │  └─ hogp.py
│  │
│  ├─coupling_gp
│  │  │  AR.py
│  │  │  CIGAR.py
│  │  │  GAR.py
│  │  │  CAR.py
│  │  │  NAR.py
│  │  └─ ResGP.py
│  │
│  ├─kernel
│  │  │  Kernel_guide.md
│  │  │  kernel_utils.py
│  │  │  MCMC_res_kernel.py
│  │  └─ SE_kernel.py
│  │
│  ├─multiscale_coupling
│  │  │  matrix.py
│  └─ └─  Residual.py
│
├─nn
├─utils
│  │  dict_tools.py
│  │  gp_noise.py
│  │  mfgp_log.py
│  │  normalizer.py
└─ └─  plot_field.py
```

# Model define rules
- For basic model, take CIGP as example.
``` python
Class CIGP(torch.nn.Mudule)
    def __init__(self, config):
        ...

    def forward(self, x, x_var):
        # this function used for predict result
        ...

    def compute_loss(self, x, y, x_var, y_var):
        # this function used for compute nll loss
        ...
```

- For coupling gp model, take AR as example.
``` python
Class AR(torch.nn.Mudule)
    def __init__(self, config):
        ...

    def single_fidelity_forward(x, low_fidelity_y, x_var=0., low_fidelity_y_var=0., fidelity_index=0):
        # Used to predict the high fidelity result from x, low_fidelity_y
        ...

    def single_fidelity_compute_loss(self, x, low_fidelity, high_fidelity_y, x_var=0., low_fidelity_var=0., high_fidelity_y_var=0., fidelity_index=0):
        # Used to train each fidelity model seperately.
        ...

    def forward(self, x, x_var, to_fidelity_n=-1):
        # Used to predict fidelity_n result from x
        # to_fidelity_n = -1 means get the final fidelity result
        ...

    def compute_loss(self, x, y_list, to_fidelity_n=-1):
        # Used to train multi-stage fidelity at one epoch
        # to_fidelity_n = -1 means training the whole model at one epoch
        ...
```