import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from emukit.test_functions.multi_fidelity import \
    (multi_fidelity_borehole_function,
     multi_fidelity_branin_function,
     multi_fidelity_currin_function,
     multi_fidelity_hartmann_3d,
     multi_fidelity_park_function,
     )

## colected data 
from assets.MF_data import \
(
    multi_fidelity_Colville, multi_fidelity_non_linear_sin, multi_fidelity_forrester_my, multi_fidelity_Toal,

    multi_fidelity_maolin1, multi_fidelity_maolin5, multi_fidelity_maolin6, multi_fidelity_maolin7, multi_fidelity_maolin8, multi_fidelity_maolin10,
    multi_fidelity_maolin12, multi_fidelity_maolin13, multi_fidelity_maolin15, multi_fidelity_maolin19, multi_fidelity_maolin20,

    multi_fidelity_p1_simp, multi_fidelity_p2_simp, multi_fidelity_p3_simp, multi_fidelity_p4_simp, multi_fidelity_p5_simp,

    multi_fidelity_shuo6, multi_fidelity_shuo11, multi_fidelity_shuo15, multi_fidelity_shuo16,

    multi_fidelity_test3_function, multi_fidelity_test4_function, multi_fidelity_test5_function, multi_fidelity_test6_function,
    multi_fidelity_test7_function, multi_fidelity_test8_function, multi_fidelity_test9_function,

    test_function_d1, test_function_d2, test_function_d3, test_function_d4, test_function_d5,
    test_function_d6, test_function_d7, test_function_d8, test_function_d9, test_function_d10,
)

from emukit.core import ContinuousParameter, ParameterSpace
from emukit.core.initial_designs.latin_design import LatinDesign
from collections import namedtuple
import torch
import numpy as np

def load_data(seed, data_name, n_train, n_test, x_normal=False, y_normal=False):

    torch.manual_seed(seed)
    np.random.seed(seed)


    if data_name == "borehole":
        fcn, space = multi_fidelity_borehole_function()

    elif data_name == "branin":
        fcn, space = multi_fidelity_branin_function()

    elif data_name == "currin":
        fcn, space = multi_fidelity_currin_function()

    elif data_name == "park":
        fcn, space = multi_fidelity_park_function()

    elif data_name == "hartmann":
        fcn, space = multi_fidelity_hartmann_3d()

    elif data_name == "colville":
        fcn, space = multi_fidelity_Colville()

    elif data_name == "nonlinearsin":
        fcn, space = multi_fidelity_non_linear_sin()

    elif data_name == "toal":
        fcn, space = multi_fidelity_Toal()

    elif data_name == "forrester":
        fcn, space = multi_fidelity_forrester_my()

    elif data_name == "p1":
        fcn, space = multi_fidelity_p1_simp()

    elif data_name == "p2":
        fcn, space = multi_fidelity_p2_simp()

    elif data_name == "p3":
        fcn, space = multi_fidelity_p3_simp()

    elif data_name == "p4":
        fcn, space = multi_fidelity_p4_simp()

    elif data_name == "p5":
        fcn, space = multi_fidelity_p5_simp()


    elif data_name == "maolin1":
        fcn, space = multi_fidelity_maolin1()

    elif data_name == "maolin5":
        fcn, space = multi_fidelity_maolin5()

    elif data_name == "maolin6":
        fcn, space = multi_fidelity_maolin6()

    elif data_name == "maolin7":
        fcn, space = multi_fidelity_maolin7()

    elif data_name == "maolin8":
        fcn, space = multi_fidelity_maolin8()

    elif data_name == "maolin10":
        fcn, space = multi_fidelity_maolin10()

    elif data_name == "maolin12":
        fcn, space = multi_fidelity_maolin12()

    elif data_name == "maolin13":
        fcn, space = multi_fidelity_maolin13()

    elif data_name == "maolin15":
        fcn, space = multi_fidelity_maolin15()

    elif data_name == "maolin19":
        fcn, space = multi_fidelity_maolin19()

    elif data_name == "maolin20":
        fcn, space = multi_fidelity_maolin20()

    elif data_name == "tl1":
        fcn, space = test_function_d1()

    elif data_name == "tl2":
        fcn, space = test_function_d2()

    elif data_name == "tl3":
        fcn, space = test_function_d3()

    elif data_name == "tl4":
        fcn, space = test_function_d4()

    elif data_name == "tl5":
        fcn, space = test_function_d5()

    elif data_name == "tl6":
        fcn, space = test_function_d6()

    elif data_name == "tl7":
        fcn, space = test_function_d7()

    elif data_name == "tl8":
        fcn, space = test_function_d8()

    elif data_name == "tl9":
        fcn, space = test_function_d9()

    elif data_name == "tl10":
        fcn, space = test_function_d10()


    elif data_name == "shuo6":
        fcn, space = multi_fidelity_shuo6()

    elif data_name == "shuo11":
        fcn, space = multi_fidelity_shuo11()

    elif data_name == "shuo15":
        fcn, space = multi_fidelity_shuo15()

    elif data_name == "shuo16":
        fcn, space = multi_fidelity_shuo16()


    elif data_name == "test3":
        fcn, space = multi_fidelity_test3_function()

    elif data_name == "test4":
        fcn, space = multi_fidelity_test4_function()

    elif data_name == "test5":
        fcn, space = multi_fidelity_test5_function()

    elif data_name == "test6":
        fcn, space = multi_fidelity_test6_function()

    elif data_name == "test7":
        fcn, space = multi_fidelity_test7_function()

    elif data_name == "test8":
        fcn, space = multi_fidelity_test8_function()

    elif data_name == "test9":
        fcn, space = multi_fidelity_test9_function()


    if data_name == "branin":
        new_space = ParameterSpace([ContinuousParameter('x1', -5., 0.), ContinuousParameter('x2', 10., 15.)])
    else:
        new_space = ParameterSpace(space._parameters[:-1])

    total_fidelity_num = len(fcn.f)
    latin = LatinDesign(new_space)

    # generate new train data
    xtr = latin.get_samples(n_train)
    Ytr = []
    for i in range(total_fidelity_num):
        Ytr.append(fcn.f[i](xtr))

    # generate new test data
    xte = latin.get_samples(n_test)
    Yte = []
    for i in range(total_fidelity_num):
        Yte.append(fcn.f[i](xte))

    # normalize X data
    if x_normal == True:
        xtr_mean, xtr_std = xtr.mean(axis=0), xtr.std(axis=0)
        xtr = (xtr - xtr_mean) / xtr_std
        xte = (xte - xtr_mean) / xtr_std

    # normalize Y data
    if y_normal == True:
        for i in range(total_fidelity_num):
            ytr_mean, ytr_std = Ytr[i].mean(axis=0), Ytr[i].std(axis=0)
            Ytr[i] = (Ytr[i] - ytr_mean) / ytr_std
            Yte[i] = (Yte[i] - ytr_mean) / ytr_std

    return xtr, Ytr, xte, Yte


def data_mu_std(seed, data_name, n_train):

    torch.manual_seed(seed)
    np.random.seed(seed)

    if data_name == "borehole":
        fcn, space = multi_fidelity_borehole_function()
        new_space = ParameterSpace(space._parameters[:-1])
    elif data_name == "branin":
        fcn, space = multi_fidelity_branin_function()
        new_space = ParameterSpace([ContinuousParameter('x1', -5., 0.), ContinuousParameter('x2', 10., 15.)])
    elif data_name == "currin":
        fcn, space = multi_fidelity_currin_function()
        new_space = ParameterSpace(space._parameters[:-1])
    elif data_name == "park":
        fcn, space = multi_fidelity_park_function()
        new_space = ParameterSpace(space._parameters[:-1])
    elif data_name == "hartmann":
        fcn, space = multi_fidelity_hartmann_3d()
        new_space = ParameterSpace(space._parameters[:-1])

    elif data_name == "tl1":
        fcn, space = test_function_d1()
        new_space = ParameterSpace(space._parameters[:-1])

    elif data_name == "tl2":
        fcn, space = test_function_d2()
        new_space = ParameterSpace(space._parameters[:-1])

    elif data_name == "tl3":
        fcn, space = test_function_d3()
        new_space = ParameterSpace(space._parameters[:-1])

    elif data_name == "tl4":
        fcn, space = test_function_d4()
        new_space = ParameterSpace(space._parameters[:-1])

    elif data_name == "tl5":
        fcn, space = test_function_d5()
        new_space = ParameterSpace(space._parameters[:-1])

    elif data_name == "tl6":
        fcn, space = test_function_d6()
        new_space = ParameterSpace(space._parameters[:-1])
    elif data_name == "tl7":
        fcn, space = test_function_d7()
        new_space = ParameterSpace(space._parameters[:-1])

    elif data_name == "tl8":
        fcn, space = test_function_d8()
        new_space = ParameterSpace(space._parameters[:-1])

    elif data_name == "tl9":
        fcn, space = test_function_d9()
        new_space = ParameterSpace(space._parameters[:-1])

    elif data_name == "tl10":
        fcn, space = test_function_d10()
        new_space = ParameterSpace(space._parameters[:-1])

    total_fidelity_num = len(fcn.f)
    latin = LatinDesign(new_space)

    # generate new train data
    xtr = latin.get_samples(n_train)
    Ytr = []

    for i in range(total_fidelity_num):
        Ytr.append(fcn.f[i](xtr))

    xtr_mean, xtr_std = xtr.mean(axis=0), xtr.std(axis=0)

    ytr_f_mean = []
    ytr_f_std = []

    for i in range(total_fidelity_num):
        ytr_mean = Ytr[i].mean(axis=0)
        ytr_std = Ytr[i].std(axis=0)
        ytr_f_mean.append(ytr_mean)
        ytr_f_std.append(ytr_std)

    return [xtr_mean, xtr_std], ytr_f_mean, ytr_f_std


def data_mu_std_v2(seed, data_name, n_train):
    """

    add function of choosing fidelity

    """
    torch.manual_seed(seed)
    np.random.seed(seed)

    xtr, Ytr, xte, Yte = load_data_certain_fi(seed=seed, data_name_with_fi=data_name, n_train=n_train, n_test=100)

    xtr_mean, xtr_std = xtr.mean(axis=0), xtr.std(axis=0)

    ytr_f_mean = []
    ytr_f_std = []

    for i in range(2):
        ytr_mean = Ytr[i].mean(axis=0)
        ytr_std = Ytr[i].std(axis=0)
        ytr_f_mean.append(ytr_mean)
        ytr_f_std.append(ytr_std)

    return [xtr_mean, xtr_std], ytr_f_mean, ytr_f_std


def load_data_v2(seed, data_name, n_train, n_test, x_normal=False, y_normal=False):
    """
    V2: add more data
    Parameters

    """
    torch.manual_seed(seed)
    np.random.seed(seed)

    fcn = None
    space = None

    if data_name == "borehole":
        fcn, space = multi_fidelity_borehole_function()

    elif data_name == "branin":
        fcn, space = multi_fidelity_branin_function()

    elif data_name == "currin":
        fcn, space = multi_fidelity_currin_function()

    elif data_name == "park":
        fcn, space = multi_fidelity_park_function()

    elif data_name == "hartmann":
        fcn, space = multi_fidelity_hartmann_3d()

    elif data_name == "colville":
        fcn, space = multi_fidelity_Colville()

    elif data_name == "nonlinearsin":
        fcn, space = multi_fidelity_non_linear_sin()

    elif data_name == "toal":
        fcn, space = multi_fidelity_Toal()

    elif data_name == "forrester":
        fcn, space = multi_fidelity_forrester_my()

    elif data_name == "p1":
        fcn, space = multi_fidelity_p1_simp()

    elif data_name == "p2":
        fcn, space = multi_fidelity_p2_simp()

    elif data_name == "p3":
        fcn, space = multi_fidelity_p3_simp()

    elif data_name == "p4":
        fcn, space = multi_fidelity_p4_simp()

    elif data_name == "p5":
        fcn, space = multi_fidelity_p5_simp()


    elif data_name == "maolin1":
        fcn, space = multi_fidelity_maolin1()

    elif data_name == "maolin5":
        fcn, space = multi_fidelity_maolin5()

    elif data_name == "maolin6":
        fcn, space = multi_fidelity_maolin6()

    elif data_name == "maolin7":
        fcn, space = multi_fidelity_maolin7()

    elif data_name == "maolin8":
        fcn, space = multi_fidelity_maolin8()

    elif data_name == "maolin10":
        fcn, space = multi_fidelity_maolin10()

    elif data_name == "maolin12":
        fcn, space = multi_fidelity_maolin12()

    elif data_name == "maolin13":
        fcn, space = multi_fidelity_maolin13()

    elif data_name == "maolin15":
        fcn, space = multi_fidelity_maolin15()

    elif data_name == "maolin19":
        fcn, space = multi_fidelity_maolin19()

    elif data_name == "maolin20":
        fcn, space = multi_fidelity_maolin20()


    elif data_name == "tl1":
        fcn, space = test_function_d1()

    elif data_name == "tl2":
        fcn, space = test_function_d2()

    elif data_name == "tl3":
        fcn, space = test_function_d3()

    elif data_name == "tl4":
        fcn, space = test_function_d4()

    elif data_name == "tl5":
        fcn, space = test_function_d5()

    elif data_name == "tl6":
        fcn, space = test_function_d6()

    elif data_name == "tl7":
        fcn, space = test_function_d7()

    elif data_name == "tl8":
        fcn, space = test_function_d8()

    elif data_name == "tl9":
        fcn, space = test_function_d9()

    elif data_name == "tl10":
        fcn, space = test_function_d10()


    elif data_name == "shuo6":
        fcn, space = multi_fidelity_shuo6()

    elif data_name == "shuo11":
        fcn, space = multi_fidelity_shuo11()

    elif data_name == "shuo15":
        fcn, space = multi_fidelity_shuo15()

    elif data_name == "shuo16":
        fcn, space = multi_fidelity_shuo16()


    elif data_name == "test3":
        fcn, space = multi_fidelity_test3_function()

    elif data_name == "test4":
        fcn, space = multi_fidelity_test4_function()

    elif data_name == "test5":
        fcn, space = multi_fidelity_test5_function()

    elif data_name == "test6":
        fcn, space = multi_fidelity_test6_function()

    elif data_name == "test7":
        fcn, space = multi_fidelity_test7_function()

    elif data_name == "test8":
        fcn, space = multi_fidelity_test8_function()

    elif data_name == "test9":
        fcn, space = multi_fidelity_test9_function()


    if data_name == "branin":
        new_space = ParameterSpace([ContinuousParameter('x1', -5., 0.), ContinuousParameter('x2', 10., 15.)])
    else:
        new_space = ParameterSpace(space._parameters[:-1])


    total_fidelity_num = len(fcn.f)
    latin = LatinDesign(new_space)

    # generate new train data
    xtr = latin.get_samples(n_train)
    Ytr = []
    for i in range(total_fidelity_num):
        Ytr.append(fcn.f[i](xtr))

    # generate new test data
    xte = latin.get_samples(n_test)
    Yte = []
    for i in range(total_fidelity_num):
        Yte.append(fcn.f[i](xte))

    # normalize X data
    if x_normal == True:
        xtr_mean, xtr_std = xtr.mean(axis=0), xtr.std(axis=0)
        xtr = (xtr - xtr_mean) / xtr_std
        xte = (xte - xtr_mean) / xtr_std

    # normalize Y data
    if y_normal == True:
        for i in range(total_fidelity_num):
            ytr_mean, ytr_std = Ytr[i].mean(axis=0), Ytr[i].std(axis=0)
            Ytr[i] = (Ytr[i] - ytr_mean) / ytr_std
            Yte[i] = (Yte[i] - ytr_mean) / ytr_std

    return xtr, Ytr, xte, Yte


def load_data_certain_fi(seed, data_name_with_fi, n_train, n_test, x_normal=False, y_normal=False):

    """
        load data with certain fidelity, coupled with load_data_v2
    """

    data_name = data_name_with_fi[:-2]

    f_high_idx = int(data_name_with_fi[-1]) - 1
    f_low_idx = int(data_name_with_fi[-2]) - 1

    xtr, Ytr_list, xte, Yte_list = load_data_v2(seed, data_name, n_train, n_test, x_normal, y_normal)

    Ytr = [Ytr_list[f_low_idx], Ytr_list[f_high_idx]]
    Yte = [Yte_list[f_low_idx], Yte_list[f_high_idx]]

    return xtr, Ytr, xte, Yte


def get_full_name_list_with_fidelity(data_name_list):
    """
        get data name list with detail fidelity
    """

    data_name_list_new = []

    for name in data_name_list:
        xtr, Ytr, xte, Yte = load_data_v2(seed=0, data_name=name, n_train=100, n_test=100, x_normal=True, y_normal=True)

        fi_len = len(Ytr)

        if fi_len == 2:
            name_with_fi = name + "12"
            data_name_list_new.append(name_with_fi)
        elif fi_len == 3:
            for fi in ["12", "13", "23"]:
                name_with_fi = name + fi
                data_name_list_new.append(name_with_fi)
        elif fi_len == 4:
            for fi in ["12", "13", "14", "23", "24", "34"]:
                name_with_fi = name + fi
                data_name_list_new.append(name_with_fi)
        else:
            print("[ error! ]")
            assert fi_len == 2

    print(data_name_list_new)
    return data_name_list_new


if __name__ == "__main__":

    # all_data_name_list = ["borehole", "branin", "currin", "park", "hartmann",
    #                       "colville", "nonlinearsin", "toal", "forrester",
    #                       "tl1", "tl2", "tl3", "tl4", "tl5", "tl6", "tl7", "tl8", "tl9", "tl10",
    #                       "p1", "p2", "p3", "p4", "p5",
    #                       "maolin1", "maolin5", "maolin6", "maolin7", "maolin8", "maolin10", "maolin12", "maolin13",
    #                       "maolin15",
    #                       "maolin19", "maolin20",
    #                       "shuo6", "shuo11", "shuo15", "shuo16",
    #                       "test3", "test4", "test5", "test6", "test7", "test8", "test9", ]

    all_data_name_with_fi_list = ['borehole12', 'branin12', 'branin13', 'branin23', 'currin12', 'park12', 'hartmann12', 'hartmann13', 'hartmann23',
                         'colville12', 'nonlinearsin12', 'toal12', 'forrester12', 'forrester13', 'forrester14', 'forrester23',
                         'forrester24', 'forrester34', 'tl112', 'tl212', 'tl312', 'tl412', 'tl512', 'tl612', 'tl712', 'tl812', 'tl912',
                         'tl1012', 'p112', 'p113', 'p123', 'p212', 'p213', 'p223', 'p312', 'p313', 'p323', 'p412', 'p413', 'p423', 'p512',
                         'p513', 'p523', 'maolin112', 'maolin512', 'maolin612', 'maolin712', 'maolin812', 'maolin1012', 'maolin1212',
                         'maolin1312', 'maolin1512', 'maolin1912', 'maolin2012', 'shuo612', 'shuo1112', 'shuo1512', 'shuo1612', 'test312',
                         'test412', 'test512', 'test612', 'test712', 'test812', 'test912']



    # get_full_name_list_with_fidelity(data_name_list=all_data_name_list)


    for data_name in all_data_name_with_fi_list:
        xtr, Ytr, xte, Yte = load_data_certain_fi(seed=0, data_name_with_fi=data_name, n_train=100, n_test=100, x_normal=True, y_normal=True)

        print(f"[ {data_name} ]: xtr_shape:{xtr.shape}")
        for i,y in enumerate(Ytr):
            print(f"Ytr{i+1}_shape: {y.shape}")

        for i, y in enumerate(Yte):
            print(f"Yte{i + 1}_shape: {y.shape}")

        print(" ")

    pass