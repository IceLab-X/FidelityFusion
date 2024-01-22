from typing import Tuple
import numpy as np
from emukit.core.loop.user_function import MultiSourceFunctionWrapper
from emukit.core import ContinuousParameter, InformationSourceParameter, ParameterSpace

PI = 3.14159

def multi_fidelity_forrester_my(std=0) -> Tuple[MultiSourceFunctionWrapper, ParameterSpace]:

    space = ParameterSpace([ContinuousParameter('x', 0, 1), InformationSourceParameter(2)])

    x_dim = 1

    def forrester_1(x, sd=std):
        """
            .. math::
        f(x) = (6x - 2)^2 \sin(12x - 4)
        """

        x = x.reshape((len(x), 1))
        n = x.shape[0]
        fval = ((6 * x - 2) ** 2) * np.sin(12 * x - 4)
        if sd == 0:
            noise = np.zeros(n).reshape(n, 1)
        else:
            noise = np.random.normal(0, sd, n).reshape(n, 1)
        return fval.reshape(n, 1) + noise


    def forrester_2(x, sd=std):
        x = x.reshape((len(x), 1))
        n = x.shape[0]
        fval = ((5.5 * x - 2.5) ** 2) * np.sin(12 * x - 4)
        if sd == 0:
            noise = np.zeros(n).reshape(n, 1)
        else:
            noise = np.random.normal(0, sd, n).reshape(n, 1)
        return fval.reshape(n, 1) + noise


    def forrester_3(x, sd=std):
        high_fidelity = forrester_1(x, 0)
        return 0.75 * high_fidelity + 5 * (x[:, [0]] - 0.5) - 2 + np.random.randn(x.shape[0], 1) * sd


    def forrester_4(x, sd=std):
        """
            .. math::
        f_{low}(x) = 0.5 f_{high}(x) + 10 (x - 0.5) - 5
        """
        high_fidelity = forrester_1(x, 0)
        return 0.5 * high_fidelity + 10 * (x[:, [0]] - 0.5) - 5 + np.random.randn(x.shape[0], 1) * sd

    return MultiSourceFunctionWrapper([forrester_4, forrester_3, forrester_2, forrester_1]), space

def multi_fidelity_non_linear_sin(high_fidelity_noise_std_deviation=0, low_fidelity_noise_std_deviation=0):
    """
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

    parameter_space = ParameterSpace([ContinuousParameter('x1', -5, 10), InformationSourceParameter(2)])
    user_function = MultiSourceFunctionWrapper([
        lambda x: nonlinear_sin_low(x, low_fidelity_noise_std_deviation),
        lambda x: nonlinear_sin_high(x, high_fidelity_noise_std_deviation)])
    return user_function, parameter_space

def nonlinear_sin_low(x, sd=0):
    """
    Low fidelity version of nonlinear sin function
    """

    return np.sin(8 * np.pi * x) + np.random.randn(x.shape[0], 1) * sd

def nonlinear_sin_high(x, sd=0):

    """
    High fidelity version of nonlinear sin function
    """

    return (x - np.sqrt(2)) * nonlinear_sin_low(x, 0) ** 2 + np.random.randn(x.shape[0], 1) * sd

def multi_fidelity_Colville(A=0.5) -> Tuple[MultiSourceFunctionWrapper, ParameterSpace]:

    space = ParameterSpace([ContinuousParameter('x1', -1., 1.), ContinuousParameter('x2', -1., 1.),
                            ContinuousParameter('x3', -1., 1.), ContinuousParameter('x4', -1., 1.),
                            InformationSourceParameter(2)])
    
    x_dim = 4

    def test_high(x):
        x1 = x[:, 0]
        x2 = x[:, 1]
        x3 = x[:, 2]
        x4 = x[:, 3]
        # x_list = [x1, x2]

        return ( 100*(x1**2-x2)**2 + (x1-1)**2 + (x3-1)**2 + 90*(x3**2-x4)
                 + 10.1*((x2-1)**2+(x4-1)**2) + 19.8*(x2-1)*(x4-1) )[:, None]

    def test_low(x):
        x1 = x[:, 0]
        x2 = x[:, 1]
        x3 = x[:, 2]
        x4 = x[:, 3]
        # x_list = [x1, x2]

        high = test_high(A * A * x).flatten()

        return ( high - (A+0.5) * (5*x1**2 + 4*x2**2 + 3*x3**2 + x4**2) )[:, None]

    return MultiSourceFunctionWrapper([test_low, test_high]), space

def test_function_d1() -> Tuple[MultiSourceFunctionWrapper, ParameterSpace]:
    def high(x):
        return (6*x-2)**2*np.sin(12*x-4)
    
    def low(x):
        return 0.56*((6*x-2)**2*np.sin(12*x-4))+10*(x-0.5)-5
    
    
    space = ParameterSpace([ContinuousParameter('x1', .0, 1.0), InformationSourceParameter(1)])
    return MultiSourceFunctionWrapper([low, high]), space 

def test_function_d2() -> Tuple[MultiSourceFunctionWrapper, ParameterSpace]:
    def high(x):
        return np.sin(2*np.pi*(x-0.1))+x**2
    
    def low(x):
        return np.sin(2*np.pi*(x-0.1))
    space = ParameterSpace([ContinuousParameter('x1', 0, 1), InformationSourceParameter(1)])
    return MultiSourceFunctionWrapper([low, high]), space  

def test_function_d3() -> Tuple[MultiSourceFunctionWrapper, ParameterSpace]:
    def high(x):
        return x*np.sin(x)/10
    
    def low(x):
        return x*np.sin(x)/10+x/10
    
    space = ParameterSpace([ContinuousParameter('x1', .0, 10), InformationSourceParameter(1)])
    return MultiSourceFunctionWrapper([low, high]), space

def test_function_d4() -> Tuple[MultiSourceFunctionWrapper, ParameterSpace]:
    def high(x):
        return np.cos(3.5*np.pi*x)*np.exp(-1.4*x)
    
    def low(x):
        return np.cos(3.5*np.pi*x)*np.exp(-1.4*x)+0.75*x**2
    
    space = ParameterSpace([ContinuousParameter('x1', .0, 1), InformationSourceParameter(1)])
    return MultiSourceFunctionWrapper([low, high]), space

def test_function_d5() -> Tuple[MultiSourceFunctionWrapper, ParameterSpace]:
    def high(x):
        x1=x[:,0]
        x2=x[:,1]
        s= 4*x1**2-2.1*x1**4+1/3*x1**6+x1*x2-4*x2**2+4*x2**4
        return s[:,None]
    
    def low(x):
        x1=x[:,0]
        x2=x[:,1]
        s= 2*x1**2-2.1*x1**4+1/3*x1**6+0.5*x1*x2-4*x2**2+2*x2**4
        return s[:,None]
    
    space = ParameterSpace([ContinuousParameter('x1', -2, 2), ContinuousParameter('x2', -2, 2),
                            InformationSourceParameter(2)])
    return MultiSourceFunctionWrapper([low, high]), space

def test_function_d6() -> Tuple[MultiSourceFunctionWrapper, ParameterSpace]:
    def high(x):
        x1=x[:,0]
        x2=x[:,1]
        s= 1/6*((30+5*x1*np.sin(5*x1))*(4+np.exp(-5*x2))-100)
        return s[:,None]
    
    def low(x):
        x1=x[:,0]
        x2=x[:,1]
        s= 1/6*((30+5*x1*np.sin(5*x1))*(4+2/5*np.exp(-5*x2))-100)
        return s[:,None]
    
    space = ParameterSpace([ContinuousParameter('x1', 0, 1), ContinuousParameter('x2', 0, 1),
                            InformationSourceParameter(2)])
    return MultiSourceFunctionWrapper([low, high]), space

def test_function_d7() -> Tuple[MultiSourceFunctionWrapper, ParameterSpace]:
    def high(x):
        x1=x[:,0]
        x2=x[:,1]
        s= x1**4+x2**4-16*x1**2-16*x2**2+5*x1+5*x2
        return s[:,None]
    
    def low(x):
        x1=x[:,0]
        x2=x[:,1]
        s= x1**4+x2**4-16*x1**2-16*x2**2
        return s[:,None]
    
    space = ParameterSpace([ContinuousParameter('x1', -3, 4), ContinuousParameter('x2', -3, 4),
                            InformationSourceParameter(2)])
    return MultiSourceFunctionWrapper([low, high]), space

def test_function_d8() -> Tuple[MultiSourceFunctionWrapper, ParameterSpace]:
    def high(x):
        x1=x[:,0]
        x2=x[:,1]
        s=(1-2*x1+0.05*np.sin(4*np.pi*x2-x1))**2+(x2-0.5*np.sin(2*np.pi*x1))**2
        return s[:,None]
    
    def low(x):
        x1=x[:,0]
        x2=x[:,1]
        s=(1-2*x1+0.05*np.sin(4*np.pi*x2-x1))**2+4*(x2-0.5*np.sin(2*np.pi*x1))**2
        return s[:,None]
    
    space = ParameterSpace([ContinuousParameter('x1', 0, 1), ContinuousParameter('x2', 0, 1),
                            InformationSourceParameter(2)])
    return MultiSourceFunctionWrapper([low, high]), space

def test_function_d9() -> Tuple[MultiSourceFunctionWrapper, ParameterSpace]:
    def high(x):
        x1=x[:,0]
        x2=x[:,1]
        x3=x[:,2]
        s= (x1-1)**2+(x1-x2)**2+x2*x3+0.5
        return s[:,None]
    
    def low(x):
        x1=x[:,0]
        x2=x[:,1]
        x3=x[:,2]
        s= 0.2*((x1-1)**2+(x1-x2)**2+x2*x3+0.5) -0.5*x1-0.2*x1*x2-0.1
        return s[:,None]
    
    space = ParameterSpace([ContinuousParameter('x1', 0, 1), ContinuousParameter('x2', 0, 1),
                            ContinuousParameter('x3', 0, 1),
                            InformationSourceParameter(3)])
    return MultiSourceFunctionWrapper([low, high]), space

def test_function_d10() -> Tuple[MultiSourceFunctionWrapper, ParameterSpace]:
    def high(x):
        Sum=0
        for i in range(8):
            Sum+=x[:,i]**4-16*x[:,i]**2+5*x[:,i]
        return Sum[:,None]
    
    def low(x):
        Sum=0
        for i in range(8):
            Sum+=0.3*x[:,i]**4-16*x[:,i]**2+5*x[:,i]
        return Sum[:,None]
    
    space = ParameterSpace([ContinuousParameter('x1', -3, 3), 
                            ContinuousParameter('x2', -3, 3),
                            ContinuousParameter('x3', -3, 3),
                            ContinuousParameter('x4', -3, 3),
                            ContinuousParameter('x5', -3, 3),
                            ContinuousParameter('x6', -3, 3),
                            ContinuousParameter('x7', -3, 3),
                            ContinuousParameter('x8', -3, 3),
                            InformationSourceParameter(8)])
    return MultiSourceFunctionWrapper([low, high]), space

def multi_fidelity_test3_function() -> Tuple[MultiSourceFunctionWrapper, ParameterSpace]:
    r"""
    Reference:

    R. Tuo, P. Z. Qian, and C. J. Wu, “Comment: A brownian motion model for stochastic simulation with tunable precision,” *Technometrics*, vol. 55, no. 1, pp. 29–31, 2013
    """

    def test_low(x):
        x1 = x[:, 0]

        return ( np.exp(1.4 * x1) * np.cos(3.5 * np.pi *x1) )[:, None]


    def test_high(x):
        x1 = x[:, 0]

        return ( np.exp(x1) * np.cos(x1) + 1 / (x1**2) )[:, None]


    space = ParameterSpace([ContinuousParameter('x1', 0., 1.),
                            InformationSourceParameter(2)])

    return MultiSourceFunctionWrapper([test_low, test_high]), space

def multi_fidelity_test4_function() -> Tuple[MultiSourceFunctionWrapper, ParameterSpace]:
    r"""
    Reference:
    [33]D. Higdon, “Space and space-time modeling using process convolutions,” in *Quantitative methods for current environmental issues*.Springer, 2002, pp. 37–56.
    """

    def test_low(x):
        x1 = x[:, 0]
        return ( np.sin(2*np.pi*x1/10) + 0.2*np.sin(2*np.pi*x1/2.5) )[:, None]


    def test_high(x):
        x1 = x[:, 0]
        return ( np.sin(2*np.pi*x1/2.5) + np.cos(2*np.pi*x1/2.5) )[:, None]

    space = ParameterSpace([ContinuousParameter('x1', 0., 10.),
                            InformationSourceParameter(2)])

    return MultiSourceFunctionWrapper([test_low, test_high]), space

def multi_fidelity_test5_function() -> Tuple[MultiSourceFunctionWrapper, ParameterSpace]:
    r"""
    Reference:
    [34]X. Cai, H. Qiu, L. Gao, and X. Shao, “Metamodeling for high dimensional design problems by multi-fifidelity simulations,” *Structural and**Multidisciplinary Optimization*, vol. 56, no. 1, pp. 151–166, 2017.
    """

    def test_low(x):
        high = test_high(0.7 * x).flatten()
        x1 = x[:, 0]
        x2 = x[:, 1]
        return ( high + x1*x2 - 65 )[:, None]


    def test_high(x):
        x1 = x[:, 0]
        x2 = x[:, 1]
        return ( 4*(x1**2) - 2.1*(x1**4) + (x1**6)/3 - 4*(x2**2) + 4*x2**4 + x1*x2 )[:, None]

    space = ParameterSpace([ContinuousParameter('x1', -2., 2.), ContinuousParameter('x2', -2., 2.),
                            InformationSourceParameter(2)])

    return MultiSourceFunctionWrapper([test_low, test_high]), space

def multi_fidelity_test6_function() -> Tuple[MultiSourceFunctionWrapper, ParameterSpace]:
    r"""
    Reference:
    [35] R. B. Gramacy and H. K. Lee, “Adaptive design and analysis of supercomputer experiments,” *Technometrics*, vol. 51, no. 2, pp. 130–145, 2009.
    """

    def test_low(x):
        x1 = x[:, 0]
        x2 = x[:, 1]
        x3 = x[:, 2]
        x4 = x[:, 3]
        x5 = x[:, 4]
        x6 = x[:, 5]
        return ( 100*np.exp(np.sin(x1)) + 5*x2*x3 + x4 + np.exp(x5*x6) )[:, None]


    def test_high(x):
        x1 = x[:, 0]
        x2 = x[:, 1]
        x3 = x[:, 2]
        x4 = x[:, 3]
        x5 = x[:, 4]
        x6 = x[:, 5]
        return ( np.exp( np.sin( (0.9*x1+0.9*0.48)**10 ) ) + x2*x3 + x4 )[:, None]

    space = ParameterSpace([ContinuousParameter('x1', 0., 1.),ContinuousParameter('x2', 0., 1.),ContinuousParameter('x3', 0., 1.),
                            ContinuousParameter('x4', 0., 1.),ContinuousParameter('x5', 0., 1.),ContinuousParameter('x6', 0., 1.),
                            InformationSourceParameter(2)])

    return MultiSourceFunctionWrapper([test_low, test_high]), space

def multi_fidelity_test7_function() -> Tuple[MultiSourceFunctionWrapper, ParameterSpace]:
    r"""
    Reference:
    [36]J. An and A. Owen, “Quasi-regression,” *Journal of complexity*, vol. 17, no. 4, pp. 588–607, 2001.
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
            res = res + Xs[i]*np.cos(x4_sum) + Xs[i]*np.sin(x4_sum)

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
            res_cos = res_cos + Xs[i]*np.cos(x4_sum)

        res_sin = 0
        for i in range(4,8):
            res_sin = res_sin + Xs[i]*np.sin(x4_sum)

        return ( (res_sin**2 + res_cos**2)**0.5 )[:, None]

    space = ParameterSpace([ContinuousParameter('x1', 0., 2.*np.pi),ContinuousParameter('x2', 0., 2.*np.pi),ContinuousParameter('x3', 0., 2.*np.pi),ContinuousParameter('x4', 0., 2.*np.pi),
                            ContinuousParameter('x5', 0., 1.),ContinuousParameter('x6', 0., 1.),ContinuousParameter('x7', 0., 1.),ContinuousParameter('x8', 0., 1.),
                            InformationSourceParameter(2)])

    return MultiSourceFunctionWrapper([test_low, test_high]), space

def multi_fidelity_test8_function() -> Tuple[MultiSourceFunctionWrapper, ParameterSpace]:
    r"""
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


    space_list = []
    for i in range(1,21):
        space_list.append(ContinuousParameter('x'+str(i), -3., 3.))
    space_list.append(InformationSourceParameter(2))

    space = ParameterSpace(space_list)
    # space = ParameterSpace([ContinuousParameter('x1', 0., 2.*np.pi),ContinuousParameter('x2', 0., 2.*np.pi),ContinuousParameter('x3', 0., 2.*np.pi),ContinuousParameter('x4', 0., 2.*np.pi),
    #                         ContinuousParameter('x5', 0., 1.),ContinuousParameter('x6', 0., 1.),ContinuousParameter('x7', 0., 1.),ContinuousParameter('x8', 0., 1.),
    #                         InformationSourceParameter(2)])

    return MultiSourceFunctionWrapper([test_low, test_high]), space

def multi_fidelity_test9_function() -> Tuple[MultiSourceFunctionWrapper, ParameterSpace]:
    r"""
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


    space_list = []
    for i in range(1,31):
        space_list.append(ContinuousParameter('x'+str(i), -3., 2.))
    space_list.append(InformationSourceParameter(2))

    space = ParameterSpace(space_list)
    # space = ParameterSpace([ContinuousParameter('x1', 0., 2.*np.pi),ContinuousParameter('x2', 0., 2.*np.pi),ContinuousParameter('x3', 0., 2.*np.pi),ContinuousParameter('x4', 0., 2.*np.pi),
    #                         ContinuousParameter('x5', 0., 1.),ContinuousParameter('x6', 0., 1.),ContinuousParameter('x7', 0., 1.),ContinuousParameter('x8', 0., 1.),
    #                         InformationSourceParameter(2)])

    return MultiSourceFunctionWrapper([test_low, test_high]), space

def multi_fidelity_p1_simp(A=0) -> Tuple[MultiSourceFunctionWrapper, ParameterSpace]:

    def sigmoid1(x):
        return 1 / ( 1 + np.exp( 32*(x+0.5)) )

    def test_high(x):
        x1 = x[:, 0]
        sum = np.sin(30*((x1-0.9)**4)) * np.cos(2*(x1-0.9)) + (x1-0.9)/2

        R = np.max(sum) - np.min(sum)
        noise = np.random.normal(loc=0, scale=A*R, size=sum.shape[0])
        sum = sum + noise * sigmoid1(x1)

        return ( sum )[:, None]

    def test_mid(x):
        x1 = x[:, 0]
        high = test_high(x).flatten()
        sum = ( high - 1 + x1 ) / ( 1 + 0.25*x1 )

        R = np.max(sum) - np.min(sum)
        noise = np.random.normal(loc=0, scale=A*R, size=sum.shape[0])
        sum = sum + noise * sigmoid1(x1)

        return (sum)[:, None]

    def test_low(x):
        x1 = x[:, 0]
        sum = np.sin(20*((x1-0.87)**4)) * np.cos(2*(x1-0.87)) + (x1-0.87)/2 - (2.5-(0.7*x1-0.14)**2) + 2*x1

        R = np.max(sum) - np.min(sum)
        noise = np.random.normal(loc=0, scale=A*R, size=sum.shape[0])
        sum = sum + noise * sigmoid1(x1)

        return ( sum )[:, None]

    space = ParameterSpace([ContinuousParameter('x1', 0., 1.),
                            InformationSourceParameter(2)])

    return MultiSourceFunctionWrapper([test_low, test_mid, test_high]), space

def multi_fidelity_p2_simp(A=0) -> Tuple[MultiSourceFunctionWrapper, ParameterSpace]:

    def sigmoid2(x):
        return 1 / ( 1 + np.exp( -32*(x+0.5)) )

    def test_high(x):
        x1 = x[:, 0]
        sum = np.sin(30*((x1-0.9)**4)) * np.cos(2*(x1-0.9)) + (x1-0.9)/2

        R = np.max(sum) - np.min(sum)
        noise = np.random.normal(loc=0, scale=A*R, size=sum.shape[0])
        sum = sum + noise * sigmoid2(x1)

        return ( sum )[:, None]

    def test_mid(x):
        x1 = x[:, 0]
        high = test_high(x).flatten()
        sum = ( high - 1 + x1 ) / ( 1 + 0.25*x1 )

        R = np.max(sum) - np.min(sum)
        noise = np.random.normal(loc=0, scale=A*R, size=sum.shape[0])
        sum = sum + noise * sigmoid2(x1)

        return (sum)[:, None]

    def test_low(x):
        x1 = x[:, 0]
        sum = np.sin(20*((x1-0.87)**4)) * np.cos(2*(x1-0.87)) + (x1-0.87)/2 - (2.5-(0.7*x1-0.14)**2) + 2*x1

        R = np.max(sum) - np.min(sum)
        noise = np.random.normal(loc=0, scale=A*R, size=sum.shape[0])
        sum = sum + noise * sigmoid2(x1)

        return ( sum )[:, None]

    space = ParameterSpace([ContinuousParameter('x1', 0., 1.),
                            InformationSourceParameter(2)])

    return MultiSourceFunctionWrapper([test_low, test_mid, test_high]), space

def multi_fidelity_p3_simp(A=0) -> Tuple[MultiSourceFunctionWrapper, ParameterSpace]:

    def sigmoid1(x):
        return 1 / ( 1 + np.exp( 32*(x+0.5)) )

    def test_high(x):
        x1 = x[:, 0]
        x2 = x[:, 1]
        sum = 100*((x2 - x1**2)**2) + (1-x1)**2
        R = np.max(sum) - np.min(sum)
        noise = np.random.normal(loc=0, scale=A*R, size=sum.shape[0])
        sum = sum + noise * sigmoid1(x1)

        return ( sum )[:, None]

    def test_mid(x):
        x1 = x[:, 0]
        x2 = x[:, 1]
        sum = 50*((x2 - x1**2)**2) + (-2-x1)**2 - 0.5*(x1+x2)

        R = np.max(sum) - np.min(sum)
        noise = np.random.normal(loc=0, scale=A*R, size=sum.shape[0])
        sum = sum + noise * sigmoid1(x1)

        return ( sum )[:, None]

    def test_low(x):
        x1 = x[:, 0]
        x2 = x[:, 1]
        high = test_high(x).flatten()
        sum = (high-4-0.5*(x1+x2))/(10+0.25*(x1+x2))

        R = np.max(sum) - np.min(sum)
        noise = np.random.normal(loc=0, scale=A*R, size=sum.shape[0])
        sum = sum + noise * sigmoid1(x1)

        return ( sum )[:, None]

    space = ParameterSpace([ContinuousParameter('x1', -2., 2.), ContinuousParameter('x2', -2., 2.),
                            InformationSourceParameter(2)])

    return MultiSourceFunctionWrapper([test_low, test_mid, test_high]), space

def multi_fidelity_p4_simp(A=0) -> Tuple[MultiSourceFunctionWrapper, ParameterSpace]:

    space = ParameterSpace([ContinuousParameter('x1', -6., 5.), ContinuousParameter('x2', -6., 5.),
                            InformationSourceParameter(2)])
    
    x_dim = 2

    def sigmoid1(x):
        return 1 / ( 1 + np.exp( 32*(x+0.5)) )

    def test_high(x):
        x1 = x[:, 0]
        x2 = x[:, 1]

        sum = (x1**2+x2**2)/25 - np.cos(x1)*np.cos(x2/(2**0.5)) + 1
        R = np.max(sum) - np.min(sum)
        noise = np.random.normal(loc=0, scale=A*R, size=sum.shape[0])
        sum = sum + noise * sigmoid1(x1)

        return ( sum )[:, None]

    def test_mid(x):

        x1 = x[:, 0]
        x2 = x[:, 1]
        sum = np.cos(x1)*np.cos(x2/(2**0.5)) + 1
        R = np.max(sum) - np.min(sum)
        noise = np.random.normal(loc=0, scale=A*R, size=sum.shape[0])
        sum = sum + noise * sigmoid1(x1)

        return ( sum )[:, None]

    def test_low(x):

        x1 = x[:, 0]
        x2 = x[:, 1]

        sum = (x1**2+x2**2)/20 - np.cos(x1/(2**0.5)) * np.cos(x2/(3**0.5)) - 1
        R = np.max(sum) - np.min(sum)
        noise = np.random.normal(loc=0, scale=A*R, size=sum.shape[0])
        sum = sum + noise * sigmoid1(x1)

        return ( sum )[:, None]

    return MultiSourceFunctionWrapper([test_low, test_mid, test_high]), space

def multi_fidelity_p5_simp(A=0) -> Tuple[MultiSourceFunctionWrapper, ParameterSpace]:

    space = ParameterSpace([ContinuousParameter('x1', -0.1, 0.2), ContinuousParameter('x2', -0.1, 0.2),
                            InformationSourceParameter(2)])
    
    x_dim = 2

    def sigmoid3(x):
        return 1 / ( 1 + np.exp( -128*(x-0.05)) )

    def theta(fai):
        return 1-0.0001*fai

    def error(x, fai):
        x1 = x[:, 0]
        x2 = x[:, 1]
        x_list = [x1, x2]
        sum = 0
        thetares = theta(fai)
        for x_now in x_list:
            sum = sum + thetares * np.cos( 10*PI*thetares*x_now + 0.5*PI*thetares + PI )**2

        return sum

    def test_1(x):
        x1 = x[:, 0]
        x2 = x[:, 1]
        x_list = [x1, x2]

        sum = 0
        for x_now in x_list:
            sum = sum + x_now**2 + 1 - np.cos(10*PI*x_now)

        R = np.max(sum) - np.min(sum)
        noise = np.random.normal(loc=0, scale=A*R, size=sum.shape[0])
        sum = sum + noise * sigmoid3(x1)

        return ( sum )[:, None]

    def test_2(x):
        x1 = x[:, 0]
        x2 = x[:, 1]

        high = test_1(x).flatten()
        err = error(x, fai=5000)
        sum = high + err

        R = np.max(sum) - np.min(sum)
        noise = np.random.normal(loc=0, scale=A*R, size=sum.shape[0])
        sum = sum + noise * sigmoid3(x1)

        return (sum)[:, None]

    def test_3(x):
        x1 = x[:, 0]
        x2 = x[:, 1]

        high = test_1(x).flatten()
        err = error(x, fai=2500)
        sum = high + err

        R = np.max(sum) - np.min(sum)
        noise = np.random.normal(loc=0, scale=A*R, size=sum.shape[0])
        sum = sum + noise * sigmoid3(x1)

        return (sum)[:, None]

    return MultiSourceFunctionWrapper([test_3, test_2, test_1]), space

def multi_fidelity_maolin1() -> Tuple[MultiSourceFunctionWrapper, ParameterSpace]:

    space = ParameterSpace([ContinuousParameter('x1', 0., 1.),
                            InformationSourceParameter(2)])
    
    x_dim = 1

    def test_high(z):
        x1 = z[:, 0]
        return ( np.sin(10*PI*x1) / (2*x1) + (x1-1)**4 )[:, None]

    def test_low(z):
        x1 = z[:, 0]
        return ( np.sin(10*PI*x1) / (x1) + 2*(x1-1)**4 )[:, None]

    return MultiSourceFunctionWrapper([test_low, test_high]), space

def multi_fidelity_maolin5() -> Tuple[MultiSourceFunctionWrapper, ParameterSpace]:

    space = ParameterSpace([ContinuousParameter('x1', 0., 5.), ContinuousParameter('x2', 0., 5.),
                            InformationSourceParameter(2)])
    
    x_dim = 2

    def test_high(z):
        x1 = z[:, 0]
        x2 = z[:, 1]

        return ( (x2 - (5.1*x1**2)/(4*PI**2) + 5.1*x1/PI - 6) + 10*(1-0.125*PI)*np.cos(x1) )[:, None]

    def test_low(z):
        x1 = z[:, 0]
        x2 = z[:, 1]

        return ( (1-0.125*PI) * np.cos(x1) )[:, None]

    return MultiSourceFunctionWrapper([test_low, test_high]), space


"""
maolin6不确定!
"""
def multi_fidelity_maolin6() -> Tuple[MultiSourceFunctionWrapper, ParameterSpace]:
    # not sure.

    space = ParameterSpace([ContinuousParameter('x1', -1., 1.), ContinuousParameter('x2', -1., 1.),
                            InformationSourceParameter(2)])
    
    x_dim = 2

    def test_high(z):
        x1 = z[:, 0]
        x2 = z[:, 1]

        return ( 101*x1**2 + 101 * (x1**2 + x2**2)**2 )[:, None]

    def test_low(z):
        x1 = z[:, 0]
        x2 = z[:, 1]

        return ( x1**2 + 100 * (x1**2 + x2**2)**4 )[:, None]

    return MultiSourceFunctionWrapper([test_low, test_high]), space

def multi_fidelity_maolin7() -> Tuple[MultiSourceFunctionWrapper, ParameterSpace]:

    space = ParameterSpace([ContinuousParameter('x1', -5., 10.), ContinuousParameter('x2', -5., 10.),
                            InformationSourceParameter(2)])
    
    x_dim = 2

    def test_high(z):
        x1 = z[:, 0]
        x2 = z[:, 1]

        return ( (1-0.2*x2+0.05*np.sin(4*PI*x2-x1))**2 + (x2-0.5*np.sin(2*PI*x1))**2 )[:, None]

    def test_low(z):
        x1 = z[:, 0]
        x2 = z[:, 1]

        return ( (1-0.2*x2+0.05*np.sin(4*PI*x2-x1))**2 + 4*(x2-0.5*np.sin(2*PI*x1))**2 )[:, None]

    return MultiSourceFunctionWrapper([test_low, test_high]), space

def multi_fidelity_maolin8() -> Tuple[MultiSourceFunctionWrapper, ParameterSpace]:

    space = ParameterSpace([ContinuousParameter('x1', 0., 1.), ContinuousParameter('x2', 0., 1.),
                            InformationSourceParameter(2)])
    
    x_dim = 2

    def test_high(z):
        x1 = z[:, 0]
        x2 = z[:, 1]

        return ( (1.5-x1+x1*x2)**2 + (2.25-x1+x1*x2**2)**2 + (2.625-x1+x1*x2**3)**2 )[:, None]

    def test_low(z):
        x1 = z[:, 0]
        x2 = z[:, 1]

        return ( (1.5-x1+x1*x2)**2 + x1 + x2 )[:, None]

    return MultiSourceFunctionWrapper([test_low, test_high]), space

def multi_fidelity_maolin10() -> Tuple[MultiSourceFunctionWrapper, ParameterSpace]:

    space = ParameterSpace([ContinuousParameter('x1', 0., 0.5), ContinuousParameter('x2', 0., 0.5),
                            InformationSourceParameter(2)])
    
    x_dim = 2

    def test_high(z):
        x1 = z[:, 0]
        x2 = z[:, 1]

        return ( (1-np.exp(-0.5/x2)) * (2300*x1**3 + 1900*x1**2 + 2092*x2 + 60)/(100*x1**3 + 500*x1**2 + 4*x2 + 20) )[:, None]

    def test_low(z):
        x1 = z[:, 0]
        x2 = z[:, 1]

        high1 = test_high(z+0.05).flatten()

        x1_new = (x1 + 0.05).reshape([-1, 1])
        x2_new = (x2 - 0.05).reshape([-1, 1])
        x2_new[x2_new<0] = 0
        z_new = np.hstack([x1_new, x2_new])

        high2 = test_high(z_new).flatten()

        x1_new = (x1 - 0.05).reshape([-1, 1])
        x2_new = (x2 + 0.05).reshape([-1, 1])
        z_new = np.hstack([x1_new, x2_new])

        high3 = test_high(z_new).flatten()

        x1_new = (x1 - 0.05).reshape([-1, 1])
        x2_new = (x2 - 0.05).reshape([-1, 1])
        x2_new[x2_new<0] = 0
        z_new = np.hstack([x1_new, x2_new])

        high4 = test_high(z_new).flatten()

        return ( -0.4 * high1 + (high2 + high3 + high4) / 4 )[:, None]

    return MultiSourceFunctionWrapper([test_low, test_high]), space

def multi_fidelity_maolin12() -> Tuple[MultiSourceFunctionWrapper, ParameterSpace]:

    space = ParameterSpace([ContinuousParameter('x1', -2., 2.), ContinuousParameter('x2', -2., 2.),
                            InformationSourceParameter(2)])
    
    x_dim = 2

    def test_high(z):
        x1 = z[:, 0]
        x2 = z[:, 1]

        return ( x1 * np.exp(-x1**2-x2**2) )[:, None]

    def test_low(z):
        x1 = z[:, 0]
        x2 = z[:, 1]

        return ( x1 * np.exp(-x1**2-x2**2) + x1/10 )[:, None]


    return MultiSourceFunctionWrapper([test_low, test_high]), space

def multi_fidelity_maolin13() -> Tuple[MultiSourceFunctionWrapper, ParameterSpace]:

    space = ParameterSpace([ContinuousParameter('x1', -1., 1.), ContinuousParameter('x2', -1., 1.),
                            InformationSourceParameter(2)])
    
    x_dim = 2

    def test_high(z):
        x1 = z[:, 0]
        x2 = z[:, 1]

        return ( np.exp(x1+x2) * np.cos(x1*x2) )[:, None]

    def test_low(z):
        x1 = z[:, 0]
        x2 = z[:, 1]

        return ( np.exp(x1+x2) * np.cos(x1*x2) + np.cos(x1**2+x2**2) )[:, None]

    return MultiSourceFunctionWrapper([test_low, test_high]), space

def multi_fidelity_maolin15() -> Tuple[MultiSourceFunctionWrapper, ParameterSpace]:

    space = ParameterSpace([ContinuousParameter('x1', 0., 1.), ContinuousParameter('x2', 0., 1.),
                            ContinuousParameter('x3', 0., 1.), InformationSourceParameter(2)])
    
    x_dim = 3

    def test_high(z):
        x1 = z[:, 0]
        x2 = z[:, 1]
        x3 = z[:, 2]

        return ( 100 * ( np.exp(-2/(x1**1.75)) + np.exp(-2/(x2**1.75)) + np.exp(-2/(x3**1.75)) ) )[:, None]

    def test_low(z):
        x1 = z[:, 0]
        x2 = z[:, 1]
        x3 = z[:, 2]

        return ( 100 * ( np.exp(-2/(x1**1.75)) + np.exp(-2/(x2**1.75)) + 0.2*np.exp(-2/(x3**1.75)) ) )[:, None]

    return MultiSourceFunctionWrapper([test_low, test_high]), space

def multi_fidelity_maolin19() -> Tuple[MultiSourceFunctionWrapper, ParameterSpace]:

    space = ParameterSpace([ContinuousParameter('x1', -5., 10.), ContinuousParameter('x2', -5., 10.),
                            ContinuousParameter('x3', -5., 10.), ContinuousParameter('x4', -5., 10.),
                            ContinuousParameter('x5', -5., 10.), ContinuousParameter('x6', -5., 10.),
                            InformationSourceParameter(2)])
    
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

    return MultiSourceFunctionWrapper([test_low, test_high]), space

def multi_fidelity_maolin20() -> Tuple[MultiSourceFunctionWrapper, ParameterSpace]:

    space = ParameterSpace([ContinuousParameter('x1', 0., 1.), ContinuousParameter('x2', 0., 1.),
                            ContinuousParameter('x3', 0., 1.), ContinuousParameter('x4', 0., 1.),
                            ContinuousParameter('x5', 0., 1.), ContinuousParameter('x6', 0., 1.),
                            ContinuousParameter('x7', 0., 1.), ContinuousParameter('x8', 0., 1.),
                            InformationSourceParameter(2)])
    
    x_dim = 8

    def test_high(z):
        x1 = z[:, 0]
        x2 = z[:, 1]
        x3 = z[:, 2]

        sum = 0
        for i in [3,4,5,6,7]:
            sum_temp = 0

            for j in np.arange(2,i+1):
                x_j = z[:, j]
                sum_temp = sum_temp + x_j

            sum = sum + (i+1) * np.log(1+sum_temp)

        sum = sum * 16 * ((x3+1)**0.5) * (2*x3-1)**2

        return ( 4*(x1-2+8*x2+8*x2**2)**2 + (3-4*x2)**2 +sum )[:, None]

    def test_low(z):
        x1 = z[:, 0]
        x2 = z[:, 1]
        x3 = z[:, 2]

        sum = 0
        for i in [3,4,5,6,7]:
            sum_temp = 0

            for j in np.arange(2,i+1):
                x_j = z[:, j]
                sum_temp = sum_temp + x_j

            sum = sum + np.log(1+sum_temp)

        sum = sum * 16 * ((x3+1)**0.5) * ((2*x3-1)**2)

        return ( 4*(x1-2+8*x2+8*x2**2)**2 + (3-4*x2)**2 + sum )[:, None]

    return MultiSourceFunctionWrapper([test_low, test_high]), space

def multi_fidelity_Toal(A=0.5) -> Tuple[MultiSourceFunctionWrapper, ParameterSpace]:

    space = ParameterSpace([ContinuousParameter('x1', -100., 100.), ContinuousParameter('x2', -100., 100.),
                            ContinuousParameter('x3', -100., 100.), ContinuousParameter('x4', -100., 100.),
                            ContinuousParameter('x5', -100., 100.), ContinuousParameter('x6', -100., 100.),
                            ContinuousParameter('x7', -100., 100.), ContinuousParameter('x8', -100., 100.),
                            ContinuousParameter('x9', -100., 100.), ContinuousParameter('x10', -100., 100.),
                            InformationSourceParameter(2)])
    
    x_dim = 10

    def test_high(z):

        sum1 = 0

        for i in range(x_dim):
            x_now = z[:, i]
            sum1 = sum1 + (x_now-1)**2

        sum2 = 0

        for i in np.arange(1, x_dim):
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

        for i in np.arange(1, x_dim):
            x_now = z[:, i]
            x_last = z[:, i-1]
            sum2 = sum2 + x_last * x_now * i * (A-0.65)

        return (sum1-sum2)[:, None]

    return MultiSourceFunctionWrapper([test_low, test_high]), space

def multi_fidelity_shuo6() -> Tuple[MultiSourceFunctionWrapper, ParameterSpace]:

    space = ParameterSpace([ContinuousParameter('x1', -5., 10.), ContinuousParameter('x2', 0., 15.),
                            InformationSourceParameter(2)])
    
    x_dim = 2

    def test_high(z):
        x1 = z[:, 0]
        x2 = z[:, 1]

        term1 = (x2 - 1.275 * (x1/PI)**2 + 5*x1/PI - 6)**2
        term2 = 10 * (1 - 1/8/PI) * np.cos(x1)

        return ( term1 + term2 )[:, None]

    def test_low(z):
        x1 = z[:, 0]
        x2 = z[:, 1]

        term1 = 0.5 * (x2 - 1.275 * (x1/PI)**2 + 5*x1/PI - 6)**2
        term2 = 10 * (1 - 1/8/PI) * np.cos(x1)

        return ( term1 + term2 )[:, None]

    return MultiSourceFunctionWrapper([test_low, test_high]), space

def multi_fidelity_shuo11() -> Tuple[MultiSourceFunctionWrapper, ParameterSpace]:

    space = ParameterSpace([ContinuousParameter('x1', -1., 1.), ContinuousParameter('x2', -1., 1.),
                            ContinuousParameter('x3', -1., 1.), InformationSourceParameter(2)])
    
    x_dim = 3

    def test_high(z):
        x1 = z[:, 0]
        x2 = z[:, 1]
        x3 = z[:, 2]

        sum = 0
        for i in range(x_dim):
            xi = z[:, i]
            sum = sum + 0.3 * np.sin(16/15*xi-1) + ( np.sin(16/15*xi-1) )**2

        return ( sum )[:, None]

    def test_low(z):
        x1 = z[:, 0]
        x2 = z[:, 1]
        x3 = z[:, 2]

        sum = 0
        for i in range(x_dim):
            xi = z[:, i]
            sum = sum + 0.3 * np.sin(16/15*xi-1) + 0.2*(np.sin(16/15*xi-1))**2

        return ( sum )[:, None]

    return MultiSourceFunctionWrapper([test_low, test_high]), space

def multi_fidelity_shuo15() -> Tuple[MultiSourceFunctionWrapper, ParameterSpace]:

    space = ParameterSpace([ContinuousParameter('x1', 0., 1.), ContinuousParameter('x2', 0., 1.),
                            ContinuousParameter('x3', 0., 1.), ContinuousParameter('x4', 0., 1.),
                            ContinuousParameter('x5', 0., 1.), ContinuousParameter('x6', 0., 1.),
                            ContinuousParameter('x7', 0., 1.), ContinuousParameter('x8', 0., 1.),
                            InformationSourceParameter(2)])
    
    x_dim = 8

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

    return MultiSourceFunctionWrapper([test_low, test_high]), space

def multi_fidelity_shuo16() -> Tuple[MultiSourceFunctionWrapper, ParameterSpace]:

    space = ParameterSpace([ContinuousParameter('x1', -2., 3.), ContinuousParameter('x2', -2., 3.),
                            ContinuousParameter('x3', -2., 3.), ContinuousParameter('x4', -2., 3.),
                            ContinuousParameter('x5', -2., 3.), ContinuousParameter('x6', -2., 3.),
                            ContinuousParameter('x7', -2., 3.), ContinuousParameter('x8', -2., 3.),
                            ContinuousParameter('x9', -2., 3.), ContinuousParameter('x10', -2., 3.),
                            InformationSourceParameter(2)])
    
    x_dim = 10

    def test_high(z):
        A = [-6.089, -17.164, -34.054, -5.914, -24.721, -14.986, -24.100, -10.708, -26.662, -22.662, -22.179]

        sum = 0
        for i in range(x_dim):

            sum_temp = 0
            for k in range(x_dim):
                sum_temp = sum_temp + np.exp(z[:, k])

            sum = sum + np.exp(z[:,i]) * ( A[i] + z[:,i] - np.log(sum_temp) )

        return ( sum )[:, None]

    def test_low(z):
        B = [-10, -10, -20, -10, -20, -20, -20, -10, -20, -20]
        sum = 0
        for i in range(x_dim):

            sum_temp = 0
            for k in range(x_dim):
                sum_temp = sum_temp + np.exp(z[:, k])

            sum = sum + np.exp(z[:,i]) * ( B[i] + z[:,i] - np.log(sum_temp) )

        return ( sum )[:, None]

    return MultiSourceFunctionWrapper([test_low, test_high]), space

