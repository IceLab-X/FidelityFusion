import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import torch
import sys
import os
from assets.MF_data.collected_data import *

## push to experiment
data_mapping = {
        "colville": multi_fidelity_Colville,
        "nonlinearsin": multi_fidelity_non_linear_sin,
        "toal": multi_fidelity_Toal,
        "forrester": multi_fidelity_forrester_my,
        "p1": multi_fidelity_p1_simp,
        "p2": multi_fidelity_p2_simp,
        "p3": multi_fidelity_p3_simp,
        "p4": multi_fidelity_p4_simp,
        "p5": multi_fidelity_p5_simp,
        "maolin1": multi_fidelity_maolin1,
        "maolin5": multi_fidelity_maolin5,
        "maolin6": multi_fidelity_maolin6,
        "maolin7": multi_fidelity_maolin7,
        "maolin8": multi_fidelity_maolin8,
        "maolin10": multi_fidelity_maolin10,
        "maolin12": multi_fidelity_maolin12,
        "maolin13": multi_fidelity_maolin13,
        "maolin15": multi_fidelity_maolin15,
        "maolin19": multi_fidelity_maolin19,
        "maolin20": multi_fidelity_maolin20,
        "tl1": test_function_d1,
        "tl2": test_function_d2,
        "tl3": test_function_d3,
        "tl4": test_function_d4,
        "tl5": test_function_d5,
        "tl6": test_function_d6,
        "tl7": test_function_d7,
        "tl8": test_function_d8,
        "tl9": test_function_d9,
        "tl10": test_function_d10,
        "shuo6": multi_fidelity_shuo6,
        "shuo11": multi_fidelity_shuo11,
        "shuo15": multi_fidelity_shuo15,
        "shuo16": multi_fidelity_shuo16,
        "test3": multi_fidelity_test3_function,
        "test4": multi_fidelity_test4_function,
        "test5": multi_fidelity_test5_function,
        "test6": multi_fidelity_test6_function,
        "test7": multi_fidelity_test7_function,
        "test8": multi_fidelity_test8_function,
        "test9": multi_fidelity_test9_function,
    }

def load_data(seed, data_name, n_train, n_test, x_normal=False, y_normal=False):
    """
    Load data for training and testing.

    Args:
        seed (int): Random seed for reproducibility.
        data_name (str): Name of the data to load.
        n_train (int): Number of samples for training.
        n_test (int): Number of samples for testing.
        x_normal (bool, optional): Flag indicating whether to normalize the X data. Defaults to False.
        y_normal (bool, optional): Flag indicating whether to normalize the Y data. Defaults to False.

    Returns:
        tuple: A tuple containing the following elements:
            - xtr (tensor): Training data for X.
            - Ytr (list): Training data for Y.
            - xte (tensor): Testing data for X.
            - Yte (list): Testing data for Y.
    """

    torch.manual_seed(seed)

    data_func = data_mapping.get(data_name, None)

    if data_func is not None:
        result = data_func() # Call the function to get data and space
        x, y_all = result

    total_fidelity_num = len(y_all)

    # generate new train data
    xtr = x[:n_train]
    Ytr = []
    for i in range(total_fidelity_num):
        Ytr.append(y_all[i][:n_train])

    # generate new test data
    xte = x[n_train:n_train + n_test]
    Yte = []
    for i in range(total_fidelity_num):
        Yte.append(y_all[i][n_train:n_train + n_test])

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


def get_data_mu_std(seed, data_name, n_train):
    """
    Calculate the mean and standard deviation of the training data and fidelity means and standard deviations.

    Args:
        seed (int): The random seed for reproducibility.
        data_name (str): The name of the data.
        n_train (int): The number of training samples.

    Returns:
        tuple: A tuple containing the mean and standard deviation of the training data (xtr_mean, xtr_std),
               and the fidelity means and standard deviations (ytr_f_mean, ytr_f_std).
    """

    torch.manual_seed(seed)

    data_func = data_mapping.get(data_name, None)

    if data_func is not None:
        result = data_func()
        x, y_all = result

    total_fidelity_num = len(y_all)

    # generate new train data
    xtr = x[:n_train]
    Ytr = []

    for i in range(total_fidelity_num):
        Ytr.append(y_all[i][:n_train])

    xtr_mean, xtr_std = xtr.mean(axis=0), xtr.std(axis=0)

    ytr_f_mean = []
    ytr_f_std = []

    for i in range(total_fidelity_num):
        ytr_mean = Ytr[i].mean(axis=0)
        ytr_std = Ytr[i].std(axis=0)
        ytr_f_mean.append(ytr_mean)
        ytr_f_std.append(ytr_std)

    return xtr_mean, xtr_std, ytr_f_mean, ytr_f_std

def load_data_certain_fi(seed, data_name_with_fi, n_train, n_test, x_normal=False, y_normal=False):
    """
    Load data for a certain feature importance (fi) range.

    Args:
        seed(int): Random seed for data loading.
        data_name_with_fi (str): Name of the data with fidelity range.
        n_train (int): Number of training samples.
        n_test (int): Number of testing samples.
        x_normal (bool, optional): Flag indicating whether to normalize the input features. Defaults to False.
        y_normal (bool, optional): Flag indicating whether to normalize the output labels. Defaults to False.

    Returns:
        tuple: A tuple containing the following elements:
            - xtr (tensor): Training input features.
            - Ytr (list): List of training output labels for the specified fidelity range.
            - xte (tensor): Testing input features.
            - Yte (list): List of testing output labels for the specified fidelity range.
    """

    data_name = data_name_with_fi[:-2]

    f_high_idx = int(data_name_with_fi[-1]) - 1
    f_low_idx = int(data_name_with_fi[-2]) - 1

    xtr, Ytr_list, xte, Yte_list = load_data(seed, data_name, n_train, n_test, x_normal, y_normal)

    Ytr = [Ytr_list[f_low_idx], Ytr_list[f_high_idx]]
    Yte = [Yte_list[f_low_idx], Yte_list[f_high_idx]]

    return xtr, Ytr, xte, Yte

def get_data_std_certain_fi(seed, data_name, n_train):
    """
    Calculate the mean and standard deviation of the training data and target variables.

    Args:
        seed (int): The random seed value.
        data_name (str): The name of the data.
        n_train (int): The number of training samples.

    Returns:
        tuple: A tuple containing the mean and standard deviation of the training data and target variables.
            - xtr_mean (tensor): The mean of the training data.
            - xtr_std (tensor): The standard deviation of the training data.
            - ytr_f_mean (list): A list of means for each target variable.
            - ytr_f_std (list): A list of standard deviations for each target variable.
    """
    torch.manual_seed(seed)

    xtr, Ytr, xte, Yte = load_data_certain_fi(seed=seed, data_name_with_fi=data_name, n_train=n_train, n_test=100)

    xtr_mean, xtr_std = xtr.mean(axis=0), xtr.std(axis=0)

    ytr_f_mean = []
    ytr_f_std = []

    for i in range(2):
        ytr_mean = Ytr[i].mean(axis=0)
        ytr_std = Ytr[i].std(axis=0)
        ytr_f_mean.append(ytr_mean)
        ytr_f_std.append(ytr_std)

    return xtr_mean, xtr_std, ytr_f_mean, ytr_f_std


def get_full_name_list_with_fidelity(data_name_list):
    """
    Returns a list of data names with fidelity values appended based on the length of the input data names.

    Args:
        data_name_list (list): A list of data names.

    Returns:
        list: A list of data names with fidelity values appended.

    Raises:
        AssertionError: If the length of the input data names is not 2.

    """
    all_data_name_with_fi_list = []

    for name in data_name_list:
        xtr, Ytr, xte, Yte = load_data(seed=0, data_name=name, n_train=100, n_test=100, x_normal=True, y_normal=True)

        fi_len = len(Ytr)

        if fi_len == 2:
            name_with_fi = name + "12"
            all_data_name_with_fi_list.append(name_with_fi)
        elif fi_len == 3:
            for fi in ["12", "13", "23"]:
                name_with_fi = name + fi
                all_data_name_with_fi_list.append(name_with_fi)
        elif fi_len == 4:
            for fi in ["12", "13", "14", "23", "24", "34"]:
                name_with_fi = name + fi
                all_data_name_with_fi_list.append(name_with_fi)
        else:
            print("[ error! ]")
            assert fi_len == 2

    print(all_data_name_with_fi_list)
    return all_data_name_with_fi_list


if __name__ == "__main__":

    all_data_name_list = ["colville", "nonlinearsin", "toal", "forrester",
                          "tl1", "tl2", "tl3", "tl4", "tl5", "tl6", "tl7", "tl8", "tl9", "tl10",
                          "p1", "p2", "p3", "p4", "p5",
                          "maolin1", "maolin5", "maolin6", "maolin7", "maolin8", "maolin10", "maolin12", "maolin13",
                          "maolin15",
                          "maolin19", "maolin20",
                          "shuo6", "shuo11", "shuo15", "shuo16",
                          "test3", "test4", "test5", "test6", "test7", "test8", "test9"]
    # all_data_name_list = ["test7"]
    
    all_data_name_with_fi_list = get_full_name_list_with_fidelity(data_name_list=all_data_name_list)

    for data_name in all_data_name_with_fi_list:

        xtr, Ytr, xte, Yte = load_data_certain_fi(seed=0, data_name_with_fi=data_name, n_train=100, n_test=100, x_normal=True, y_normal=True)

        xtr_mean, xtr_std, ytr_f_mean, ytr_f_std = get_data_std_certain_fi(seed=0,data_name=data_name, n_train=100)

        print(f"[ {data_name} ]: xtr_shape:{xtr.shape}")
        for i,y in enumerate(Ytr):
            print(f"Ytr{i+1}_shape: {y.shape}")

        print(f"xte_shape:{xte.shape}")
        for i, y in enumerate(Yte):
            print(f"Yte{i + 1}_shape: {y.shape}")

        print(" ")
