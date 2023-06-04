

## Kernel Guide

All the kernel is define as follow:

```
Class kernel(torch.nn.Module):
#inheritance from torch.nn.Module, thus we can easy to do back propagation
	def __init__(self)
	#add new param if need
		...
		
	def forward(self,x1,x2)
		...

	def get_param(self)
	#return param, using to save/check nan.
```



## Refererence

doc refer: https://www.cs.toronto.edu/~duvenaud/cookbook/



## kernel list

- [x] Squared Exponential Kernel
- [x] Rational Quadratic Kernel
- [ ] Periodic Kernel
- [ ] Locally Periodic Kernel
- [x] Linear Kernel
- [x] combine Kernel
