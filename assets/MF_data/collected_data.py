import torch
import warnings

def multi_fidelity_forrester_my(x = None, min_value = 0, max_value = 1, std = 0, num_points = 200):
    '''
    x_dim = 1
    '''
    if x is None:
        x = torch.rand(num_points, 1) * (max_value - min_value) + min_value
    else:
        if not torch.all((x >= min_value) & (x <= max_value)):
            warnings.warn("Input data is out of specified range [{}, {}].".format(min_value, max_value))
    def forrester_1(x, sd=std):
        """
        math:
            f(x) = (6x - 2)^2 \sin(12x - 4)
        """
        n = x.size(0)
        fval = ((6 * x - 2)**2) * torch.sin(12 * x - 4)
        noise = torch.zeros(n, 1) if sd == 0 else torch.randn(n, 1) * sd
        return fval + noise

    def forrester_2(x, sd=std):
        """
        math:
            f(x) = (5.5x - 2.5)^2 \sin(12x - 4)
        """
        n = x.size(0)
        fval = ((5.5 * x - 2.5)**2) * torch.sin(12 * x - 4)
        noise = torch.zeros(n, 1) if sd == 0 else torch.randn(n, 1) * sd
        return fval + noise

    def forrester_3(x, sd=std):
        '''
        math:
            f(x) = 0.75 f_{1}(x) + 5 (x - 0.5) - 2
        '''
        high_fidelity = forrester_1(x, 0)
        return 0.75 * high_fidelity + 5 * (x - 0.5) - 2 + torch.randn(x.size(0), 1) * sd

    def forrester_4(x, sd=std):
        """
        math:
            f_{low}(x) = 0.5 f_{1}(x) + 10 (x - 0.5) - 5
        """
        high_fidelity = forrester_1(x, 0)
        noise = torch.randn(x.size(0), 1) * sd
        return 0.5 * high_fidelity + 10 * (x - 0.5) - 5 + noise

    return x,[forrester_1(x), forrester_2(x), forrester_3(x), forrester_4(x)]

def multi_fidelity_non_linear_sin(x = None, min_value = -5, max_value = 10, high_fidelity_noise_std_deviation=0, low_fidelity_noise_std_deviation=0, num_points = 200):
    """
    x_dim = 1

    Two level non-linear sin function where high fidelity is given by:

    .. math::
        f_{high}(x) = (x - \sqrt{2}) f_{low}(x)^2

    and the low fidelity is:

    .. math::
        f_{low}(x) = \sin(8 \pi x)

    Reference:
    Nonlinear information fusion algorithms for data-efficient multi-fidelity modelling.
    P. Perdikaris, M. Raissi, A. Damianou, N. D. Lawrence and G. E. Karniadakis (2017)
    http://web.mit.edu/parisp/www/assets/20160751.full.pdf
    """
    if x is None:
        x = torch.rand(num_points, 1) * (max_value - min_value) + min_value
    else:
        if not torch.all((x >= min_value) & (x <= max_value)):
            warnings.warn("Input data is out of specified range [{}, {}].".format(min_value, max_value))
    return x, [nonlinear_sin_low(x, low_fidelity_noise_std_deviation), nonlinear_sin_high(x, high_fidelity_noise_std_deviation)]

def nonlinear_sin_low(x, sd=0):
    """
    Low fidelity version of nonlinear sin function
    """

    return torch.sin(8 * torch.pi * x) + torch.randn(x.size(0), 1) * sd

def nonlinear_sin_high(x, sd=0):
    """
    High fidelity version of nonlinear sin function
    """

    return (x - torch.sqrt(torch.tensor(2.0))) * nonlinear_sin_low(x, 0) ** 2 + torch.randn(x.size(0), 1) * sd

def multi_fidelity_Colville(x = None, A=0.5, min_value = -1, max_value = 1, num_points = 200):
    '''
    x_dim = 4
    '''
    if x is None:
        x = (max_value - min_value) * torch.rand(num_points, 4) + min_value
    else:
        if not torch.all((x >= min_value) & (x <= max_value)):
            warnings.warn("Input data is out of specified range [{}, {}].".format(min_value, max_value))

    def test_high(x):
        '''
        math:
            f(x) = 100*(x1^2-x2)^2 + (x1-1)^2 + (x3-1)^2 + 90*(x3^2-x4) + 10.1*((x2-1)^2+(x4-1)^2) + 19.8*(x2-1)*(x4-1)
        '''
        x1 = x[:, 0]
        x2 = x[:, 1]
        x3 = x[:, 2]
        x4 = x[:, 3]
        # x_list = [x1, x2]

        return ( 100*(x1**2-x2)**2 + (x1-1)**2 + (x3-1)**2 + 90*(x3**2-x4)
                 + 10.1*((x2-1)**2+(x4-1)**2) + 19.8*(x2-1)*(x4-1) )[:, None]

    def test_low(x):
        '''
        math:
            f(x) = (A^2(x1,x2,x3,x4))*f_{low} - (A+0.5)(5*x1^2 + 4*x2^2 + 3*x3^2 + x4^2)
        '''
        x1 = x[:, 0]
        x2 = x[:, 1]
        x3 = x[:, 2]
        x4 = x[:, 3]
        # x_list = [x1, x2]

        high = test_high(A * A * x).flatten()

        return ( high - (A+0.5) * (5*x1**2 + 4*x2**2 + 3*x3**2 + x4**2) )[:, None]

    return x, [test_low(x), test_high(x)]

def test_function_d1(x = None, min_value = 0, max_value = 1, num_points = 200):
    '''
    x_dim = 1
    '''
    def high(x):
        '''
        math:
            f(x) = (6x - 2)^2 \sin(12x - 4)
        '''
        return (6 * x - 2)**2 * torch.sin(12 * x - 4)

    def low(x):
        '''
        math:
            f(x) = 0.56 * ((6x - 2)^2 \sin(12x - 4)) + 10(x - 0.5) - 5
        '''
        return 0.56 * ((6 * x - 2)**2 * torch.sin(12 * x - 4)) + 10 * (x - 0.5) - 5
    if x is None:
        x = torch.rand(num_points, 1) * (max_value - min_value) + min_value
    else:
        if not torch.all((x >= min_value) & (x <= max_value)):
            warnings.warn("Input data is out of specified range [{}, {}].".format(min_value, max_value))

    return x, [low(x), high(x)] 

def test_function_d2(x = None, min_value = 0, max_value = 1, num_points = 200):
    '''
    x_dim = 1
    '''
    def high(x):
        '''
        math:
            f(x) = \sin(2 \pi (x - 0.1)) + x^2
        '''
        return torch.sin(2 * torch.pi * (x - 0.1)) + x.pow(2)

    def low(x):
        '''
        math:
            f(x) = \sin(2 \pi (x - 0.1))
        '''
        return torch.sin(2 * torch.pi * (x - 0.1))
    
    if x is None:
        x = torch.rand(num_points, 1) * (max_value - min_value) + min_value
    else:
        if not torch.all((x >= min_value) & (x <= max_value)):
            warnings.warn("Input data is out of specified range [{}, {}].".format(min_value, max_value))
    return x, [low(x), high(x)]  

def test_function_d3(x = None, min_value = 0, max_value = 10, num_points = 200):
    '''
    x_dim = 1
    '''
    def high(x):
        '''
        math:
            f(x) = x \sin(x) / 10
        '''
        return x * torch.sin(x) / 10

    def low(x):
        '''
        math:
            f(x) = x \sin(x) / 10 + x / 10
        '''
        return x * torch.sin(x) / 10 + x / 10
    if x is None:
        x = torch.rand(num_points, 1) * (max_value - min_value) + min_value
    else:
        if not torch.all((x >= min_value) & (x <= max_value)):
            warnings.warn("Input data is out of specified range [{}, {}].".format(min_value, max_value))
    return x, [low(x), high(x)]

def test_function_d4(x = None, min_value = 0, max_value = 1, num_points = 200):
    '''
    x_dim = 1
    '''
    def high(x):
        '''
        math:
            f(x) = \cos(3.5 \pi x) \exp(-1.4 x)
        '''
        return torch.cos(3.5 * torch.pi * x) * torch.exp(-1.4 * x)

    def low(x):
        '''
        math:
            f(x) = \cos(3.5 \pi x) \exp(-1.4 x) + 0.75 x^2
        '''
        return torch.cos(3.5 * torch.pi * x) * torch.exp(-1.4 * x) + 0.75 * x ** 2
    
    if x is None:
        x = torch.rand(num_points, 1) * (max_value - min_value) + min_value
    else:
        if not torch.all((x >= min_value) & (x <= max_value)):
            warnings.warn("Input data is out of specified range [{}, {}].".format(min_value, max_value))
    
    return x, [low(x), high(x)]

def test_function_d5(x = None, min_value = -2, max_value = 2, num_points = 200):
    '''
    x_dim = 2
    '''
    def high(x):
        '''
        math:
            f(x) = 4x^2 - 2.1x^4 + 1/3x^6 + x1x2 - 4x2^2 + 4x2^4
        '''
        x1 = x[:, 0]
        x2 = x[:, 1]
        s = 4 * x1**2 - 2.1 * x1**4 + 1/3 * x1**6 + x1 * x2 - 4 * x2**2 + 4 * x2**4
        return s.unsqueeze(1)

    def low(x):
        '''
        math:
            f(x) = 2x^2 - 2.1x^4 + 1/3x^6 + 0.5x1x2 - 4x2^2 + 2x2^4
        '''
        x1 = x[:, 0]
        x2 = x[:, 1]
        s = 2 * x1**2 - 2.1 * x1**4 + 1/3 * x1**6 + 0.5 * x1 * x2 - 4 * x2**2 + 2 * x2**4
        return s.unsqueeze(1)
    if x is None:
        x = torch.rand(num_points, 2) * (max_value - min_value) + min_value
    else:
        if not torch.all((x >= min_value) & (x <= max_value)):
            warnings.warn("Input data is out of specified range [{}, {}].".format(min_value, max_value))
    return x, [low(x), high(x)]

def test_function_d6(x = None, min_value = 0, max_value = 1, num_points = 200):
    '''
    x_dim = 2
    '''
    def high(x):
        '''
        math:
            f(x) = 1/6 * ((30 + 5x1 \sin(5x1)) (4 + \exp(-5x2)) - 100)
        '''
        x1 = x[:, 0]
        x2 = x[:, 1]
        s = 1/6 * ((30 + 5 * x1 * torch.sin(5 * x1)) * (4 + torch.exp(-5 * x2)) - 100)
        return s.unsqueeze(1)

    def low(x):
        '''
        math:
            f(x) = 1/6 * ((30 + 5x1 \sin(5x1)) (4 + 2/5 \exp(-5x2)) - 100)
        '''
        x1 = x[:, 0]
        x2 = x[:, 1]
        s = 1/6 * ((30 + 5 * x1 * torch.sin(5 * x1)) * (4 + 2/5 * torch.exp(-5 * x2)) - 100)
        return s.unsqueeze(1)
    
    if x is None:
        x = torch.rand(num_points, 2) * (max_value - min_value) + min_value
    else:
        if not torch.all((x >= min_value) & (x <= max_value)):
            warnings.warn("Input data is out of specified range [{}, {}].".format(min_value, max_value))
    return x, [low(x), high(x)]

def test_function_d7(x = None, min_value = -3, max_value = 4, num_points = 200):
    '''
    x_dim = 2
    '''
    def high(x):
        '''
        math:
            f(x) = x1^4 + x2^4 - 16x1^2 - 16x2^2 + 5x1 + 5x2
        '''
        x1 = x[:, 0]
        x2 = x[:, 1]
        s = x1**4 + x2**4 - 16*x1**2 - 16*x2**2 + 5*x1 + 5*x2
        return s.unsqueeze(1)

    def low(x):
        '''
        math:
            f(x) = x1^4 + x2^4 - 16x1^2 - 16x2^2
        '''
        x1 = x[:, 0]
        x2 = x[:, 1]
        s = x1**4 + x2**4 - 16*x1**2 - 16*x2**2
        return s.unsqueeze(1)
    
    if x is None:
        x = torch.rand(num_points, 2) * (max_value - min_value) + min_value
    else:
        if not torch.all((x >= min_value) & (x <= max_value)):
            warnings.warn("Input data is out of specified range [{}, {}].".format(min_value, max_value))
    return x, [low(x), high(x)]

def test_function_d8(x = None, min_value = 0, max_value = 1, num_points = 200):
    '''
    x_dim = 2
    '''
    def high(x):
        '''
        math:
            f(x) = (1 - 2x1 + 0.05 \sin(4\pi x2 - x1))^2 + (x2 - 0.5 \sin(2\pi x1))^2
        '''
        x1 = x[:, 0]
        x2 = x[:, 1]
        s = (1 - 2*x1 + 0.05*torch.sin(4*torch.pi*x2 - x1))**2 + (x2 - 0.5*torch.sin(2*torch.pi*x1))**2
        return s.unsqueeze(1)

    def low(x):
        '''
        math:
            f(x) = (1 - 2x1 + 0.05 \sin(4\pi x2 - x1)^2 + 4*(x2 - 0.5 \sin(2\pi x1))^2)
        '''
        x1 = x[:, 0]
        x2 = x[:, 1]
        s = (1 - 2*x1 + 0.05*torch.sin(4*torch.pi*x2 - x1))**2 + 4*(x2 - 0.5*torch.sin(2*torch.pi*x1))**2
        return s.unsqueeze(1)
    
    if x is None:
        x = torch.rand(num_points, 2) * (max_value - min_value) + min_value
    else:
        if not torch.all((x >= min_value) & (x <= max_value)):
            warnings.warn("Input data is out of specified range [{}, {}].".format(min_value, max_value))
    return x, [low(x), high(x)]

def test_function_d9(x = None, min_value = 0, max_value = 1, num_points = 200):
    '''
    x_dim = 3
    '''
    def high(x):
        '''
        math:
            f(x) = (x1 - 1)^2 + (x1 - x2)^2 + x2x3 + 0.5
        '''
        x1 = x[:, 0]
        x2 = x[:, 1]
        x3 = x[:, 2]
        s = (x1 - 1)**2 + (x1 - x2)**2 + x2 * x3 + 0.5
        return s.unsqueeze(1)

    def low(x):
        '''
        math:
            f(x) = 0.2 * ((x1 - 1)^2 + (x1 - x2)^2 + x2x3 + 0.5) - 0.5x1 - 0.2x1x2 - 0.1
        '''
        x1 = x[:, 0]
        x2 = x[:, 1]
        x3 = x[:, 2]
        s = 0.2 * ((x1 - 1)**2 + (x1 - x2)**2 + x2 * x3 + 0.5) - 0.5 * x1 - 0.2 * x1 * x2 - 0.1
        return s.unsqueeze(1)
    
    if x is None:
        x = torch.rand(num_points, 3) * (max_value - min_value) + min_value
    else:
        if not torch.all((x >= min_value) & (x <= max_value)):
            warnings.warn("Input data is out of specified range [{}, {}].".format(min_value, max_value))
    return x, [low(x), high(x)]

def test_function_d10(x = None, min_value = -3, max_value = 3, num_points = 200):
    '''
    x_dim = 8
    '''
    def high(x):
        Sum = 0
        for i in range(8):
            Sum += x[:, i]**4 - 16 * x[:, i]**2 + 5 * x[:, i]
        return Sum.unsqueeze(1)

    def low(x):
        Sum = 0
        for i in range(8):
            Sum += 0.3 * x[:, i]**4 - 16 * x[:, i]**2 + 5 * x[:, i]
        return Sum.unsqueeze(1)
    
    if x is None:
        x = torch.rand(num_points, 8) * (max_value - min_value) + min_value
    else:
        if not torch.all((x >= min_value) & (x <= max_value)):
            warnings.warn("Input data is out of specified range [{}, {}].".format(min_value, max_value))
    
    return x, [low(x), high(x)]

def multi_fidelity_test3_function(x = None, min_value = 0, max_value = 1, num_points = 200):
    r"""
    x_dim = 1

    Reference:

    R. Tuo, P. Z. Qian, and C. J. Wu, “Comment: A brownian motion model for stochastic simulation with tunable precision,” *Technometrics*, vol. 55, no. 1, pp. 29–31, 2013
    """

    def test_low(x):
        '''
        math:
            f(x) = \exp(1.4x1)* \cos(3.5\pi x1)
        '''
        x1 = x[:, 0]

        return (torch.exp(1.4 * x1) * torch.cos(3.5 * torch.pi * x1))[:, None]

    def test_high(x):
        '''
        math:
            f(x) = \exp(x1)* \cos(x1) + 1/x1^2
        '''
        x1 = x[:, 0]

        return (torch.exp(x1) * torch.cos(x1) + 1 / (x1**2))[:, None]

    if x is None:
        x = torch.rand(num_points, 1) * (max_value - min_value) + min_value
    else:
        if not torch.all((x >= min_value) & (x <= max_value)):
            warnings.warn("Input data is out of specified range [{}, {}].".format(min_value, max_value))

    return x, [test_low(x), test_high(x)]

def multi_fidelity_test4_function(x = None, min_value = 0, max_value = 10, num_points = 200):
    r"""
    x_dim = 1

    Reference:
    [33]D. Higdon, “Space and space-time modeling using process convolutions,” in *Quantitative methods for current environmental issues*.Springer, 2002, pp. 37–56.
    """

    def test_low(x):
        '''
        math:
            f(x) = \sin(2 \pi x1 / 10) + 0.2 \sin(2 \pi x1 / 2.5)
        '''
        x1 = x[:, 0]
        return (torch.sin(2 * torch.pi * x1 / 10) + 0.2 * torch.sin(2 * torch.pi * x1 / 2.5))[:, None]

    def test_high(x):
        '''
        math:
            f(x) = \sin(2 \pi x1 / 2.5) + \cos(2 \pi x1 / 2.5)
        '''
        x1 = x[:, 0]
        return (torch.sin(2 * torch.pi * x1 / 2.5) + torch.cos(2 * torch.pi * x1 / 2.5))[:, None]

    if x is None:
        x = torch.rand(num_points, 1) * (max_value - min_value) + min_value
    else:
        if not torch.all((x >= min_value) & (x <= max_value)):
            warnings.warn("Input data is out of specified range [{}, {}].".format(min_value, max_value))

    return x, [test_low(x), test_high(x)]

def multi_fidelity_test5_function(x = None, min_value = -2, max_value = 2, num_points = 200):
    r"""
    x_dim = 2
    Reference:
    [34]X. Cai, H. Qiu, L. Gao, and X. Shao, “Metamodeling for high dimensional design problems by multi-fifidelity simulations,” *Structural and**Multidisciplinary Optimization*, vol. 56, no. 1, pp. 151–166, 2017.
    """

    def test_low(x):

        high = test_high(0.7 * x)
        x1 = x[:, 0]
        x2 = x[:, 1]
        return (high.flatten() + x1 * x2 - 65)[:, None]

    def test_high(x):
        '''
        math:
            f(x) = 4x1^2 - 2.1x1^4 + x1^6 / 3 - 4x2^2 + 4x2^4 + x1x2
        '''
        x1 = x[:, 0]
        x2 = x[:, 1]
        return (4 * (x1 ** 2) - 2.1 * (x1 ** 4) + (x1 ** 6) / 3 - 4 * (x2 ** 2) + 4 * x2 ** 4 + x1 * x2)[:, None]

    if x is None:
        x = torch.rand(num_points, 2) * (max_value - min_value) + min_value
    else:
        if not torch.all((x >= min_value) & (x <= max_value)):
            warnings.warn("Input data is out of specified range [{}, {}].".format(min_value, max_value))

    return x, [test_low(x), test_high(x)]

def multi_fidelity_test6_function(x = None, min_value = 0, max_value = 1, num_points = 200):
    r"""
    x_dim = 6
    Reference:
    [35] R. B. Gramacy and H. K. Lee, “Adaptive design and analysis of supercomputer experiments,” *Technometrics*, vol. 51, no. 2, pp. 130–145, 2009.
    """

    def test_low(x):
        '''
        math:
            f(x) = 100 \exp(\sin(x1)) + 5x2x3 + x4 + \exp(x5x6)
        '''
        x1 = x[:, 0]
        x2 = x[:, 1]
        x3 = x[:, 2]
        x4 = x[:, 3]
        x5 = x[:, 4]
        x6 = x[:, 5]
        return (100 * torch.exp(torch.sin(x1)) + 5 * x2 * x3 + x4 + torch.exp(x5 * x6))[:, None]

    def test_high(x):
        '''
        math:
            f(x) = \exp(\sin(0.9x1 + 0.9*0.48)^{10}) + x2x3 + x4
        '''
        x1 = x[:, 0]
        x2 = x[:, 1]
        x3 = x[:, 2]
        x4 = x[:, 3]
        return (torch.exp(torch.sin((0.9 * x1 + 0.9 * 0.48) ** 10)) + x2 * x3 + x4)[:, None]
    
    if x is None:
        x = torch.rand(num_points, 6) * (max_value - min_value) + min_value
    else:
        if not torch.all((x >= min_value) & (x <= max_value)):
            warnings.warn("Input data is out of specified range [{}, {}].".format(min_value, max_value))

    return x, [test_low(x), test_high(x)]

def multi_fidelity_test7_function(x = None, min1 = 0, max1 = 2 * torch.pi, min2 = 0, max2 = 1, num_points = 200):
    r"""
    x_dim = 8
    Reference:
    [36]J. An and A. Owen, “Quasi-regression,” *Journal of complexity*, vol. 17, no. 4, pp. 588-607, 2001.
    """

    def test_low(x):
        Xs = []
        for i in range(8):
            Xs.append(x[:,i])

        x4_sum = 0
        for i in range(4):
            x4_sum = x4_sum +Xs[i]

        res = 0
        for i in range(4,8):
            res = res + Xs[i] * torch.cos(x4_sum) + Xs[i] * torch.sin(x4_sum)

        return res[:, None]


    def test_high(x):
        Xs = []
        for i in range(8):
            Xs.append(x[:,i])

        x4_sum = 0
        for i in range(4):
            x4_sum = x4_sum +Xs[i]

        res_cos = 0
        for i in range(4,8):
            res_cos = res_cos + Xs[i] * torch.cos(x4_sum)

        res_sin = 0
        for i in range(4,8):
            res_sin = res_sin + Xs[i] * torch.sin(x4_sum)

        return ( (res_sin**2 + res_cos**2)**0.5 )[:, None]

    if x is None:
        data_list = []
        for _ in range(num_points):
            x1_to_x4 = torch.rand(4) * (max1 - min1) + min1
            x5_to_x8 = torch.rand(4) * (max2 - min2) + min2
            x = torch.cat((x1_to_x4, x5_to_x8), dim=0)
            data_list.append(x)
        x = torch.stack(data_list)
    else:
        if not torch.all((x[:, :4] >= min1) & (x[:, :4] <= max1)):
            warnings.warn("The first four dimensions of input data are out of specified range [{}, {}].".format(min1, max1))
        
        if not torch.all((x[:, 4:] >= min2) & (x[:, 4:] <= max2)):
            warnings.warn("The last four dimensions of input data are out of specified range [{}, {}].".format(min2, max2))

    return x, [test_low(x), test_high(x)]

def multi_fidelity_test8_function(x = None, min_value = -3, max_value = 3, num_points = 200):
    r"""
    x_dim = 20
    Reference:
    [20]X. Meng and G. E. Karniadakis, “A composite neural network that learns from multi-fifidelity data: Application to function approximationand inverse pde problems,” *Journal of Computational Physics*, vol. 401, p. 109020, 2020.

    """

    def test_low(x):
        x_high = test_high(x).flatten()

        Xs = []
        for i in range(20):
            Xs.append(x[:,i])

        res= 0
        for i in range(0,19):
            res = res + ( 0.4*Xs[i]*Xs[i+1] )

        return ( 0.8*x_high - res - 50 )[:, None]


    def test_high(x):
        Xs = []
        for i in range(20):
            Xs.append(x[:,i])

        res= 0
        for i in range(1,20):
            res = res + ( 2*Xs[i]**2 - Xs[i-1] )**2

        return ( res + Xs[0]**2 )[:, None]

    if x is None:
        x = torch.rand(num_points, 20) * (max_value - min_value) + min_value
    else:
        if not torch.all((x >= min_value) & (x <= max_value)):
            warnings.warn("Input data is out of specified range [{}, {}].".format(min_value, max_value))

    return x, [test_low(x), test_high(x)]

def multi_fidelity_test9_function(x = None, min_value = -3, max_value = 2, num_points = 200):
    r"""
    x_dim = 30
    Reference:
    [37]G. H. Cheng, A. Younis, K. Haji Hajikolaei, and G. Gary Wang, “Trust region based mode pursuing sampling method for global optimization of high dimensional design problems,” *Journal of Mechanical Design*, vol. 137, no. 2, 2015.
    """

    def test_low(x):
        x_high = test_high(x).flatten()

        Xs = []
        for i in range(30):
            Xs.append(x[:, i])

        return ( x_high**3 + x_high**2 + x_high )[:, None]


    def test_high(x):
        Xs = []
        for i in range(30):
            Xs.append(x[:,i])

        res= 0
        for i in range(0,29):
            res = res + (30-(i+1)) * (Xs[i]**2 - Xs[i+1])**2

        return ( (Xs[0]-1)**2 + (Xs[29]-1)**2 + 30*res )[:, None]

    if x is None:
        x = torch.rand(num_points, 30) * (max_value - min_value) + min_value
    else:
        if not torch.all((x >= min_value) & (x <= max_value)):
            warnings.warn("Input data is out of specified range [{}, {}].".format(min_value, max_value))

    return x, [test_low(x), test_high(x)]

def multi_fidelity_p1_simp(x = None, A = 0, min_value = -3, max_value = 2, num_points = 200):
    '''
    x_dim = 1
    '''
    def sigmoid1(x):
        return 1 / (1 + torch.exp(32 * (x + 0.5)))

    def test_high(x):
        x1 = x[:, 0]
        sum = torch.sin(30 * ((x1 - 0.9) ** 4)) * torch.cos(2 * (x1 - 0.9)) + (x1 - 0.9) / 2

        R = torch.max(sum) - torch.min(sum)
        noise = torch.randn_like(sum) * A * R
        sum = sum + noise * sigmoid1(x1)

        return sum[:, None]

    def test_mid(x):
        x1 = x[:, 0]
        high = test_high(x).flatten()
        sum = (high - 1 + x1) / (1 + 0.25 * x1)

        R = torch.max(sum) - torch.min(sum)
        noise = torch.randn_like(sum) * A * R
        sum = sum + noise * sigmoid1(x1)

        return sum[:, None]

    def test_low(x):
        x1 = x[:, 0]
        sum = torch.sin(20 * ((x1 - 0.87) ** 4)) * torch.cos(2 * (x1 - 0.87)) + (x1 - 0.87) / 2 - (2.5 - (0.7 * x1 - 0.14) ** 2) + 2 * x1

        R = torch.max(sum) - torch.min(sum)
        noise = torch.randn_like(sum) * A * R
        sum = sum + noise * sigmoid1(x1)

        return sum[:, None]

    if x is None:
        x = torch.rand(num_points, 1) * (max_value - min_value) + min_value
    else:
        if not torch.all((x >= min_value) & (x <= max_value)):
            warnings.warn("Input data is out of specified range [{}, {}].".format(min_value, max_value))

    return x, [test_low(x), test_mid(x), test_high(x)]

def multi_fidelity_p2_simp(x = None, A = 0, min_value = 0, max_value = 1, num_points = 200):
    '''
    x_dim = 1
    '''
    def sigmoid2(x):
        return 1 / (1 + torch.exp(-32 * (x + 0.5)))

    def test_high(x):
        x1 = x[:, 0]
        sum = torch.sin(30 * ((x1 - 0.9) ** 4)) * torch.cos(2 * (x1 - 0.9)) + (x1 - 0.9) / 2

        R = torch.max(sum) - torch.min(sum)
        noise = torch.randn_like(sum) * A * R
        sum = sum + noise * sigmoid2(x1)

        return sum[:, None]

    def test_mid(x):
        x1 = x[:, 0]
        high = test_high(x).flatten()
        sum = (high - 1 + x1) / (1 + 0.25 * x1)

        R = torch.max(sum) - torch.min(sum)
        noise = torch.randn_like(sum) * A * R
        sum = sum + noise * sigmoid2(x1)

        return sum[:, None]

    def test_low(x):
        x1 = x[:, 0]
        sum = torch.sin(20 * ((x1 - 0.87) ** 4)) * torch.cos(2 * (x1 - 0.87)) + (x1 - 0.87) / 2 - (2.5 - (0.7 * x1 - 0.14) ** 2) + 2 * x1

        R = torch.max(sum) - torch.min(sum)
        noise = torch.randn_like(sum) * A * R
        sum = sum + noise * sigmoid2(x1)

        return sum[:, None]

    if x is None:
        x = torch.rand(num_points, 1) * (max_value - min_value) + min_value
    else:
        if not torch.all((x >= min_value) & (x <= max_value)):
            warnings.warn("Input data is out of specified range [{}, {}].".format(min_value, max_value))

    return x, [test_low(x), test_mid(x), test_high(x)]

def multi_fidelity_p3_simp(x = None, A = 0, min_value = -2, max_value = 2, num_points = 200):

    '''
    x_dim = 2
    '''
    def sigmoid1(x):
        return 1 / (1 + torch.exp(32 * (x + 0.5)))

    def test_high(x):
        x1 = x[:, 0]
        x2 = x[:, 1]
        sum = 100 * ((x2 - x1**2)**2) + (1 - x1)**2
        R = torch.max(sum) - torch.min(sum)
        noise = torch.randn_like(sum) * A * R
        sum = sum + noise * sigmoid1(x1)

        return sum[:, None]

    def test_mid(x):
        x1 = x[:, 0]
        x2 = x[:, 1]
        sum = 50 * ((x2 - x1**2)**2) + (-2 - x1)**2 - 0.5 * (x1 + x2)

        R = torch.max(sum) - torch.min(sum)
        noise = torch.randn_like(sum) * A * R
        sum = sum + noise * sigmoid1(x1)

        return sum[:, None]

    def test_low(x):
        x1 = x[:, 0]
        x2 = x[:, 1]
        high = test_high(x).flatten()
        sum = (high - 4 - 0.5 * (x1 + x2)) / (10 + 0.25 * (x1 + x2))

        R = torch.max(sum) - torch.min(sum)
        noise = torch.randn_like(sum) * A * R
        sum = sum + noise * sigmoid1(x1)

        return sum[:, None]

    if x is None:
        x = torch.rand(num_points, 2) * (max_value - min_value) + min_value
    else:
        if not torch.all((x >= min_value) & (x <= max_value)):
            warnings.warn("Input data is out of specified range [{}, {}].".format(min_value, max_value))

    return x, [test_low(x), test_mid(x), test_high(x)]

def multi_fidelity_p4_simp(x = None, A = 0, min_value = -6, max_value = 5, num_points = 200):
    '''
    x_dim = 2
    '''
    def sigmoid1(x):
        return 1 / (1 + torch.exp(32 * (x + 0.5)))

    def test_high(x):
        x1 = x[:, 0]
        x2 = x[:, 1]

        sum = (x1**2 + x2**2) / 25 - torch.cos(x1) * torch.cos(x2 / (2**0.5)) + 1
        R = torch.max(sum) - torch.min(sum)
        noise = torch.randn_like(sum) * A * R
        sum = sum + noise * sigmoid1(x1)

        return sum[:, None]

    def test_mid(x):
        x1 = x[:, 0]
        x2 = x[:, 1]
        sum = torch.cos(x1) * torch.cos(x2 / (2**0.5)) + 1
        R = torch.max(sum) - torch.min(sum)
        noise = torch.randn_like(sum) * A * R
        sum = sum + noise * sigmoid1(x1)

        return sum[:, None]

    def test_low(x):
        x1 = x[:, 0]
        x2 = x[:, 1]

        sum = (x1**2 + x2**2) / 20 - torch.cos(x1 / (2**0.5)) * torch.cos(x2 / (3**0.5)) - 1
        R = torch.max(sum) - torch.min(sum)
        noise = torch.randn_like(sum) * A * R
        sum = sum + noise * sigmoid1(x1)

        return sum[:, None]

    if x is None:
        x = torch.rand(num_points, 2) * (max_value - min_value) + min_value
    else:
        if not torch.all((x >= min_value) & (x <= max_value)):
            warnings.warn("Input data is out of specified range [{}, {}].".format(min_value, max_value))

    return x, [test_low(x), test_mid(x), test_high(x)]

def multi_fidelity_p5_simp(x = None, A = 0, min_value = -0.2, max_value = -0.1, num_points = 200):
    '''
    x_dim = 2
    '''
    def sigmoid3(x):
        return 1 / (1 + torch.exp(-128 * (x - 0.05)))

    def theta(fai):
        return 1 - 0.0001 * fai

    def error(x, fai):
        x1 = x[:, 0]
        x2 = x[:, 1]
        x_list = [x1, x2]
        sum = 0
        thetares = theta(fai)
        for x_now in x_list:
            sum = sum + thetares * torch.cos(10 * torch.pi * thetares * x_now + 0.5 * torch.pi * thetares + torch.pi)**2

        return sum

    def test_1(x):
        x1 = x[:, 0]
        x2 = x[:, 1]
        x_list = [x1, x2]

        sum = 0
        for x_now in x_list:
            sum = sum + x_now**2 + 1 - torch.cos(10 * torch.pi * x_now)

        R = torch.max(sum) - torch.min(sum)
        noise = torch.randn_like(sum) * A * R
        sum = sum + noise * sigmoid3(x1)

        return sum[:, None]

    def test_2(x):
        x1 = x[:, 0]
        x2 = x[:, 1]

        high = test_1(x).flatten()
        err = error(x, fai=5000)
        sum = high + err

        R = torch.max(sum) - torch.min(sum)
        noise = torch.randn_like(sum) * A * R
        sum = sum + noise * sigmoid3(x1)

        return sum[:, None]

    def test_3(x):
        x1 = x[:, 0]
        x2 = x[:, 1]

        high = test_1(x).flatten()
        err = error(x, fai=2500)
        sum = high + err

        R = torch.max(sum) - torch.min(sum)
        noise = torch.randn_like(sum) * A * R
        sum = sum + noise * sigmoid3(x1)

        return sum[:, None]
    
    if x is None:
        x = torch.rand(num_points, 2) * (max_value - min_value) + min_value
    else:
        if not torch.all((x >= min_value) & (x <= max_value)):
            warnings.warn("Input data is out of specified range [{}, {}].".format(min_value, max_value))

    return x, [test_1(x), test_2(x), test_3(x)]

def multi_fidelity_maolin1(x = None, min_value = 0, max_value = 1, num_points = 200):
    '''
    x_dim = 1
    '''
    def test_high(z):
        x1 = z[:, 0]
        return ( torch.sin(10*torch.pi*x1) / (2*x1) + (x1-1)**4 )[:, None]

    def test_low(z):
        x1 = z[:, 0]
        return ( torch.sin(10*torch.pi*x1) / (x1) + 2*(x1-1)**4 )[:, None]
    
    if x is None:
        x = torch.rand(num_points, 1) * (max_value - min_value) + min_value
    else:
        if not torch.all((x >= min_value) & (x <= max_value)):
            warnings.warn("Input data is out of specified range [{}, {}].".format(min_value, max_value))

    return x, [test_low(x), test_high(x)]

def multi_fidelity_maolin5(x = None, min_value = 0, max_value = 5, num_points = 200):

    '''
    x_dim = 2
    '''
    def test_high(z):
        x1 = z[:, 0]
        x2 = z[:, 1]

        return ( (x2 - (5.1*x1**2)/(4*torch.pi**2) + 5.1*x1/torch.pi - 6) + 10*(1-0.125*torch.pi)*torch.cos(x1) )[:, None]

    def test_low(z):
        x1 = z[:, 0]

        return ( (1-0.125*torch.pi) * torch.cos(x1) )[:, None]
    
    if x is None:
        x = torch.rand(num_points, 2) * (max_value - min_value) + min_value
    else:
        if not torch.all((x >= min_value) & (x <= max_value)):
            warnings.warn("Input data is out of specified range [{}, {}].".format(min_value, max_value))

    return x, [test_low(x), test_high(x)]

def multi_fidelity_maolin6(x = None, min_value = 0, max_value = 5, num_points = 200):

    '''
    x_dim = 2
    '''
    def test_high(z):
        '''
        math:
            f(x) = 101x1^2 + 101(x1^2 + x2^2)^2
        '''
        x1 = z[:, 0]
        x2 = z[:, 1]

        return ( 101*x1**2 + 101 * (x1**2 + x2**2)**2 )[:, None]

    def test_low(z):
        '''
        math:
            f(x) = x1^2 + 100(x1^2 + x2^2)^4
        '''
        x1 = z[:, 0]
        x2 = z[:, 1]

        return ( x1**2 + 100 * (x1**2 + x2**2)**4 )[:, None]
    
    if x is None:
        x = torch.rand(num_points, 2) * (max_value - min_value) + min_value
    else:
        if not torch.all((x >= min_value) & (x <= max_value)):
            warnings.warn("Input data is out of specified range [{}, {}].".format(min_value, max_value))

    return x, [test_low(x), test_high(x)]

def multi_fidelity_maolin7(x = None, min_value = -5, max_value = 10, num_points = 200):

    '''
    x_dim = 2
    '''
    def test_high(z):
        '''
        math:
            f(x) = (1-0.2x2+0.05\sin(4\pi x2-x1))^2 + (x2-0.5\sin(2\pi x1))^2
        '''
        x1 = z[:, 0]
        x2 = z[:, 1]

        return ( (1-0.2*x2+0.05*torch.sin(4*torch.pi*x2-x1))**2 + (x2-0.5*torch.sin(2*torch.pi*x1))**2 )[:, None]

    def test_low(z):
        '''
        math:
            f(x) = (1-0.2x2+0.05\sin(4\pi x2-x1))^2 + 4(x2-0.5\sin(2\pi x1))^2
        '''
        x1 = z[:, 0]
        x2 = z[:, 1]

        return ( (1-0.2*x2+0.05*torch.sin(4*torch.pi*x2-x1))**2 + 4*(x2-0.5*torch.sin(2*torch.pi*x1))**2 )[:, None]
    
    if x is None:
        x = torch.rand(num_points, 2) * (max_value - min_value) + min_value
    else:
        if not torch.all((x >= min_value) & (x <= max_value)):
            warnings.warn("Input data is out of specified range [{}, {}].".format(min_value, max_value))

    return x, [test_low(x), test_high(x)]

def multi_fidelity_maolin8(x = None, min_value = 0, max_value = 1, num_points = 200):
    '''
    x_dim = 2
    '''
    def test_high(z):
        '''
        math:
            f(x) = (1.5-x1+x1x2)^2 + (2.25-x1+x1x2^2)^2 + (2.625-x1+x1x2^3)^2
        '''
        x1 = z[:, 0]
        x2 = z[:, 1]

        return ( (1.5-x1+x1*x2)**2 + (2.25-x1+x1*x2**2)**2 + (2.625-x1+x1*x2**3)**2 )[:, None]

    def test_low(z):
        '''
        math:
            f(x) = (1.5-x1+x1x2)^2 + (x1+x2)^2
        '''
        x1 = z[:, 0]
        x2 = z[:, 1]

        return ( (1.5-x1+x1*x2)**2 + x1 + x2 )[:, None]
    
    if x is None:
        x = torch.rand(num_points, 2) * (max_value - min_value) + min_value
    else:
        if not torch.all((x >= min_value) & (x <= max_value)):
            warnings.warn("Input data is out of specified range [{}, {}].".format(min_value, max_value))

    return x, [test_low(x), test_high(x)]

def multi_fidelity_maolin10(x = None, min_value = 0, max_value = 0.5, num_points = 200):
    '''
    x_dim = 2
    '''
    def test_high(z):
        x1 = z[:, 0]
        x2 = z[:, 1]

        return ( (1-torch.exp(-0.5/x2)) * (2300*x1**3 + 1900*x1**2 + 2092*x2 + 60)/(100*x1**3 + 500*x1**2 + 4*x2 + 20) )[:, None]

    def test_low(z):
        x1 = z[:, 0]
        x2 = z[:, 1]

        high1 = test_high(z+0.05).flatten()

        x1_new = (x1 + 0.05).reshape([-1, 1])
        x2_new = (x2 - 0.05).reshape([-1, 1])
        x2_new[x2_new<0] = 0
        z_new = torch.hstack([x1_new, x2_new])

        high2 = test_high(z_new).flatten()

        x1_new = (x1 - 0.05).reshape([-1, 1])
        x2_new = (x2 + 0.05).reshape([-1, 1])
        z_new = torch.hstack([x1_new, x2_new])

        high3 = test_high(z_new).flatten()

        x1_new = (x1 - 0.05).reshape([-1, 1])
        x2_new = (x2 - 0.05).reshape([-1, 1])
        x2_new[x2_new<0] = 0
        z_new = torch.hstack([x1_new, x2_new])

        high4 = test_high(z_new).flatten()

        return ( -0.4 * high1 + (high2 + high3 + high4) / 4 )[:, None]
    
    if x is None:
        x = torch.rand(num_points, 2) * (max_value - min_value) + min_value
    else:
        if not torch.all((x >= min_value) & (x <= max_value)):
            warnings.warn("Input data is out of specified range [{}, {}].".format(min_value, max_value))

    return x, [test_low(x), test_high(x)]

def multi_fidelity_maolin12(x = None, min_value = -2, max_value = 2, num_points = 200):
    '''
    x_dim = 2
    '''
    def test_high(z):
        '''
        math:
            f(x) = x1 \exp(-x1^2-x2^2)
        '''
        x1 = z[:, 0]
        x2 = z[:, 1]

        return ( x1 * torch.exp(-x1**2-x2**2) )[:, None]

    def test_low(z):
        '''
        math:
            f(x) = x1 \exp(-x1^2-x2^2) + x1/10
        '''
        x1 = z[:, 0]
        x2 = z[:, 1]

        return ( x1 * torch.exp(-x1**2-x2**2) + x1/10 )[:, None]

    if x is None:
        x = torch.rand(num_points, 2) * (max_value - min_value) + min_value
    else:
        if not torch.all((x >= min_value) & (x <= max_value)):
            warnings.warn("Input data is out of specified range [{}, {}].".format(min_value, max_value))
    return x, [test_low(x), test_high(x)]

def multi_fidelity_maolin13(x = None, min_value = -1, max_value = 1, num_points = 200):

    '''
    x_dim = 2
    '''
    def test_high(z):
        '''
        math:
            f(x) = \exp(x1+x2) \cos(x1x2)
        '''
        x1 = z[:, 0]
        x2 = z[:, 1]

        return ( torch.exp(x1+x2) * torch.cos(x1*x2) )[:, None]

    def test_low(z):
        '''
        math:
            f(x) = \exp(x1+x2) \cos(x1x2) + \cos(x1^2+x2^2)
        '''
        x1 = z[:, 0]
        x2 = z[:, 1]

        return ( torch.exp(x1+x2) * torch.cos(x1*x2) + torch.cos(x1**2+x2**2) )[:, None]
    
    if x is None:
        x = torch.rand(num_points, 2) * (max_value - min_value) + min_value
    else:
        if not torch.all((x >= min_value) & (x <= max_value)):
            warnings.warn("Input data is out of specified range [{}, {}].".format(min_value, max_value))

    return x, [test_low(x), test_high(x)]

def multi_fidelity_maolin15(x = None, min_value = 0, max_value = 1, num_points = 200):
    '''
    x_dim = 3
    '''
    def test_high(z):
        '''
        math:
            f(x) = 100 \exp(-2/x1^1.75) + 100 \exp(-2/x2^1.75) + 100 \exp(-2/x3^1.75)
        '''
        x1 = z[:, 0]
        x2 = z[:, 1]
        x3 = z[:, 2]

        return ( 100 * ( torch.exp(-2/(x1**1.75)) + torch.exp(-2/(x2**1.75)) + torch.exp(-2/(x3**1.75)) ) )[:, None]

    def test_low(z):
        '''
        math:
            f(x) = 100 \exp(-2/x1^1.75) + 100 \exp(-2/x2^1.75) + 0.2 \exp(-2/x3^1.75)
        '''
        x1 = z[:, 0]
        x2 = z[:, 1]
        x3 = z[:, 2]

        return ( 100 * ( torch.exp(-2/(x1**1.75)) + torch.exp(-2/(x2**1.75)) + 0.2*torch.exp(-2/(x3**1.75)) ) )[:, None]
    
    if x is None:
        x = torch.rand(num_points, 3) * (max_value - min_value) + min_value
    else:
        if not torch.all((x >= min_value) & (x <= max_value)):
            warnings.warn("Input data is out of specified range [{}, {}].".format(min_value, max_value))

    return x, [test_low(x), test_high(x)]

def multi_fidelity_maolin19(x = None, min_value = -5, max_value = 10, num_points = 200):
    '''
    x_dim = 6
    '''
    x_dim = 6

    def test_high(z):
        sum = 0
        for i in range(x_dim-1):
            x_now = z[:,i]
            x_next = z[:,i+1]
            sum = sum + 100*(x_next - x_now**2)**2 + (x_now-1)**2

        return ( sum )[:, None]

    def test_low(z):
        sum = 0
        for i in range(x_dim-1):
            x_now = z[:,i]
            x_next = z[:,i+1]
            sum = sum + 100*(x_next - x_now)**2 + 4*(x_now-1)**4

        return ( sum )[:, None]
    
    if x is None:
        x = torch.rand(num_points, x_dim) * (max_value - min_value) + min_value
    else:
        if not torch.all((x >= min_value) & (x <= max_value)):
            warnings.warn("Input data is out of specified range [{}, {}].".format(min_value, max_value))

    return x, [test_low(x), test_high(x)]

def multi_fidelity_maolin20(x = None, min_value = 0, max_value = 1, num_points = 200):
    '''
    x_dim = 8
    '''
    def test_high(z):
        x1 = z[:, 0]
        x2 = z[:, 1]
        x3 = z[:, 2]

        sum = 0
        for i in [3,4,5,6,7]:
            sum_temp = 0

            for j in torch.arange(2,i+1):
                x_j = z[:, j]
                sum_temp = sum_temp + x_j

            sum = sum + (i+1) * torch.log(1+sum_temp)

        sum = sum * 16 * ((x3+1)**0.5) * (2*x3-1)**2

        return ( 4*(x1-2+8*x2+8*x2**2)**2 + (3-4*x2)**2 +sum )[:, None]

    def test_low(z):
        x1 = z[:, 0]
        x2 = z[:, 1]
        x3 = z[:, 2]

        sum = 0
        for i in [3,4,5,6,7]:
            sum_temp = 0

            for j in torch.arange(2,i+1):
                x_j = z[:, j]
                sum_temp = sum_temp + x_j

            sum = sum + torch.log(1+sum_temp)

        sum = sum * 16 * ((x3+1)**0.5) * ((2*x3-1)**2)

        return ( 4*(x1-2+8*x2+8*x2**2)**2 + (3-4*x2)**2 + sum )[:, None]
    
    if x is None:
        x = torch.rand(num_points, 8) * (max_value - min_value) + min_value
    else:
        if not torch.all((x >= min_value) & (x <= max_value)):
            warnings.warn("Input data is out of specified range [{}, {}].".format(min_value, max_value))

    return x, [test_low(x), test_high(x)]

def multi_fidelity_Toal(x = None, A = 0.5, min_value = -100, max_value = 100, num_points = 200):
    '''
    x_dim = 10
    '''
    x_dim = 10

    def test_high(z):

        sum1 = 0

        for i in range(x_dim):
            x_now = z[:, i]
            sum1 = sum1 + (x_now-1)**2

        sum2 = 0

        for i in torch.arange(1, x_dim):
            x_now = z[:, i]
            x_last = z[:, i-1]
            sum2 = sum2 + x_last * x_now

        return (sum1-sum2)[:, None]

    def test_low(z):

        sum1 = 0

        for i in range(x_dim):
            x_now = z[:, i]
            sum1 = sum1 + (x_now-A)**2

        sum2 = 0

        for i in torch.arange(1, x_dim):
            x_now = z[:, i]
            x_last = z[:, i-1]
            sum2 = sum2 + x_last * x_now * i * (A-0.65)

        return (sum1-sum2)[:, None]
    
    if x is None:
        x = torch.rand(num_points, x_dim) * (max_value - min_value) + min_value
    else:
        if not torch.all((x >= min_value) & (x <= max_value)):
            warnings.warn("Input data is out of specified range [{}, {}].".format(min_value, max_value))

    return x, [test_low(x), test_high(x)]

def multi_fidelity_shuo6(x = None, min1 = -5, max1 = 10, min2 = 0, max2 = 15, num_points = 200):
    '''
    x_dim = 2
    '''
    def test_high(z):
        x1 = z[:, 0]
        x2 = z[:, 1]

        term1 = (x2 - 1.275 * (x1/torch.pi)**2 + 5*x1/torch.pi - 6)**2
        term2 = 10 * (1 - 1/8/torch.pi) * torch.cos(x1)

        return ( term1 + term2 )[:, None]

    def test_low(z):
        x1 = z[:, 0]
        x2 = z[:, 1]

        term1 = 0.5 * (x2 - 1.275 * (x1/torch.pi)**2 + 5*x1/torch.pi - 6)**2
        term2 = 10 * (1 - 1/8/torch.pi) * torch.cos(x1)

        return ( term1 + term2 )[:, None]
    
    if x is None:
        data_list = []
        for _ in range(num_points):
            x1 = torch.rand(1) * (max1 - min1) + min1
            x2 = torch.rand(1) * (max2 - min2) + min2
            x = torch.cat((x1, x2), dim=0)
            data_list.append(x)
        x = torch.stack(data_list)
    else:
        if not torch.all((x[:, :1] >= min1) & (x[:, :1] <= max1)):
            warnings.warn("The first dimensions of input data are out of specified range [{}, {}].".format(min1, max1))
        
        if not torch.all((x[:, 1:] >= min2) & (x[:, 1:] <= max2)):
            warnings.warn("The last dimensions of input data are out of specified range [{}, {}].".format(min2, max2))

    return x, [test_low(x), test_high(x)]

def multi_fidelity_shuo11(x = None, min_value = -1, max_value = 1, num_points = 200):
    
    '''
    x_dim = 3
    '''
    x_dim = 3

    def test_high(z):
        sum = 0
        for i in range(x_dim):
            xi = z[:, i]
            sum = sum + 0.3 * torch.sin(16/15*xi-1) + ( torch.sin(16/15*xi-1) )**2

        return ( sum )[:, None]

    def test_low(z):
        sum = 0
        for i in range(x_dim):
            xi = z[:, i]
            sum = sum + 0.3 * torch.sin(16/15*xi-1) + 0.2*(torch.sin(16/15*xi-1))**2

        return ( sum )[:, None]
    
    if x is None:
        x = torch.rand(num_points, x_dim) * (max_value - min_value) + min_value
    else:
        if not torch.all((x >= min_value) & (x <= max_value)):
            warnings.warn("Input data is out of specified range [{}, {}].".format(min_value, max_value))

    return x, [test_low(x), test_high(x)]

def multi_fidelity_shuo15(x = None, min_value = 0, max_value = 1, num_points = 200):
    '''
    x_dim = 8
    '''
    def test_high(z):

        sum = 0
        for i in [1, 2]:
            term1 = (z[:,(4*i-3)-1] + 10*z[:,(4*i-2)-1])**2
            term2 = 5*(z[:, (4*i-1)-1] - z[:, (4*i)-1])**2
            term3 = (z[:, (4*i-2)-1] - 2*z[:, (4*i-1)-1])**4
            term4 = 10*(z[:, (4*i-3)-1] - z[:, (4*i)-1])**4
            sum = sum + term1 + term2 + term3 + term4


        return ( sum )[:, None]

    def test_low(z):

        sum = 0
        for i in [1, 2]:
            term1 = (z[:,(4*i-3)-1] + 10*z[:,(4*i-2)-1])**2
            term2 = 125*(z[:, (4*i-1)-1] - z[:, (4*i)-1])**2
            term3 = (z[:, (4*i-2)-1] - 2*z[:, (4*i-1)-1])**4
            term4 = 10*(z[:, (4*i-3)-1] - z[:, (4*i)-1])**4
            sum = sum + term1 + term2 + term3 + term4

        return ( sum )[:, None]
    
    if x is None:
        x = torch.rand(num_points, 8) * (max_value - min_value) + min_value
    else:
        if not torch.all((x >= min_value) & (x <= max_value)):
            warnings.warn("Input data is out of specified range [{}, {}].".format(min_value, max_value))

    return x, [test_low(x), test_high(x)]

def multi_fidelity_shuo16(x = None, min_value = -2, max_value = 3, num_points = 200):
    '''
    x_dim = 10
    '''
    x_dim = 10

    def test_high(z):
        A = [-6.089, -17.164, -34.054, -5.914, -24.721, -14.986, -24.100, -10.708, -26.662, -22.662, -22.179]

        sum = 0
        for i in range(x_dim):

            sum_temp = 0
            for k in range(x_dim):
                sum_temp = sum_temp + torch.exp(z[:, k])

            sum = sum + torch.exp(z[:,i]) * ( A[i] + z[:,i] - torch.log(sum_temp) )

        return ( sum )[:, None]

    def test_low(z):
        B = [-10, -10, -20, -10, -20, -20, -20, -10, -20, -20]
        sum = 0
        for i in range(x_dim):

            sum_temp = 0
            for k in range(x_dim):
                sum_temp = sum_temp + torch.exp(z[:, k])

            sum = sum + torch.exp(z[:,i]) * ( B[i] + z[:,i] - torch.log(sum_temp) )

        return ( sum )[:, None]
    
    if x is None:
        x = torch.rand(num_points, x_dim) * (max_value - min_value) + min_value
    else:
        if not torch.all((x >= min_value) & (x <= max_value)):
            warnings.warn("Input data is out of specified range [{}, {}].".format(min_value, max_value))

    return x, [test_low(x), test_high(x)]

