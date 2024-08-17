'''
version 1.0 : 2023/12/28 finish init and add_data, get_data, get_overlap_input_data, get_unique_input_data
version 1.1 : 2024/1/4 finish display_fidelity_data and optimized code
version 1.2 : 2024/1/6 add get_data_by_name and get_nonsubset_data
version 1.3 : 2024/3/3 add normalize and denormalize
'''
import torch
import warnings
EPS = 1e-10
class Normalizer:
    """
    A class for normalizing and denormalizing data.

    Args:
        x (torch.Tensor): The input data tensor.
        y (torch.Tensor): The target data tensor.
        normal_x_dim (int, optional): The dimension along which to compute the mean and standard deviation for `x`. Defaults to 0.
        normal_y_mode (int, optional): The mode for normalizing `y`. 
            - 0: Normalize `y` all together.
            - 1: Normalize `y` by each dimension. Defaults to 0.
    """

    def __init__(self, x, y, normal_x_dim=0, normal_y_mode=0) -> None:
        self.x_mean = x.mean(dim=normal_x_dim)
        self.x_std = x.std(dim=normal_x_dim)

        if normal_y_mode == 0:
            self.y_mean = y.mean()
            self.y_std = y.std()
        elif normal_y_mode == 1:
            self.y_mean = y.mean(0)
            self.y_std = y.std(0)

    def normalize(self, x, y):
        """
        Normalize the input data `x` and target data `y`.

        Args:
            x (torch.Tensor): The input data tensor.
            y (torch.Tensor): The target data tensor.

        Returns:
            tuple: A tuple containing the normalized `x` and `y` tensors.
        """
        x = (x - self.x_mean.expand_as(x)) / (self.x_std.expand_as(x) + EPS)
        y = (y - self.y_mean.expand_as(y)) / (self.y_std.expand_as(y) + EPS)
        return x, y

    def normalize_x(self, x):
        """
        Normalize the input data `x`.

        Args:
            x (torch.Tensor): The input data tensor.

        Returns:
            torch.Tensor: The normalized `x` tensor.
        """
        return (x - self.x_mean.expand_as(x)) / (self.x_std.expand_as(x) + EPS)

    def denormalize(self, mean, var):
        """
        Denormalize the mean and variance.

        Args:
            mean (torch.Tensor): The mean tensor.
            var (torch.Tensor): The variance tensor.

        Returns:
            tuple: A tuple containing the denormalized mean and variance tensors.
        """
        mean = mean * self.y_std.expand_as(mean) + self.y_mean.expand_as(mean)
        var = var * (self.y_std ** 2).expand_as(var)
        return mean, var
    
class min_max_normalizer:
    """
    A class for performing min-max normalization on a given tensor.

    Args:
        tensor (Tensor): The input tensor to be normalized.
        min_value (float, optional): The minimum value of the normalized range. Defaults to 0.
        max_value (float, optional): The maximum value of the normalized range. Defaults to 1.
    """

    def __init__(self, tensor, min_value=0, max_value=1) -> None:
        self.min_value = min_value
        self.max_value = max_value
        self.min = tensor.min(dim=0, keepdim=True).values
        self.max = tensor.max(dim=0, keepdim=True).values
    
    def normalize(self, tensor):
        """
        Normalize the given tensor.

        Args:
            tensor (Tensor): The input tensor to be normalized.

        Returns:
            Tensor: The normalized tensor.
        """
        return (tensor - self.min) / (self.max - self.min) * (self.max_value - self.min_value) + self.min_value
    
    def denormalize(self, tensor):
        """
        Denormalize the given tensor.

        Args:
            tensor (Tensor): The input tensor to be denormalized.

        Returns:
            Tensor: The denormalized tensor.
        """
        return (tensor - self.min_value) / (self.max_value - self.min_value) * (self.max - self.min) + self.min

class min_max_normalizer_2:
    """
    A class for performing min-max normalization on selected columns of a given tensor.

    Args:
        tensor (Tensor): The input tensor to be normalized.
        columns (list of int, optional): The columns to be normalized. Defaults to [0, 1, 2].
        min_value (float, optional): The minimum value of the normalized range. Defaults to 0.
        max_value (float, optional): The maximum value of the normalized range. Defaults to 1.
    """

    def __init__(self, tensor, columns=[0, 1, 2], min_value=0, max_value=1) -> None:
        self.columns = columns
        self.min_value = min_value
        self.max_value = max_value
        self.min = tensor[:, columns].min(dim=0, keepdim=True).values  # 获取选定列的最小值
        self.max = tensor[:, columns].max(dim=0, keepdim=True).values  # 获取选定列的最大值
    
    def normalize(self, tensor):
        """
        Normalize the specified columns of the given tensor.

        Args:
            tensor (Tensor): The input tensor to be normalized.

        Returns:
            Tensor: The normalized tensor.
        """
        tensor_copy = tensor.clone()  # 创建 tensor 的副本
        tensor_copy[:, self.columns] = (tensor[:, self.columns] - self.min) / (self.max - self.min) * (self.max_value - self.min_value) + self.min_value
        return tensor_copy
    
    def denormalize(self, tensor):
        """
        Denormalize the specified columns of the given tensor.

        Args:
            tensor (Tensor): The input tensor to be denormalized.

        Returns:
            Tensor: The denormalized tensor.
        """
        tensor_copy = tensor.clone()  # 创建 tensor 的副本
        tensor_copy[:, self.columns] = (tensor[:, self.columns] - self.min_value) / (self.max_value - self.min_value) * (self.max - self.min) + self.min
        return tensor_copy

# TODO: doest data manager assume the low fidelity data always contains more data than the high fidelity data?
class MultiFidelityDataManager:
    """
    A class for managing multi-fidelity data.

    Attributes:
    - data_dict (dict): A dictionary to store the fidelity data.
    - normalizelayer (dict): A dictionary to store the normalization layers for each fidelity index.

    Methods:
    - __init__(self, initial_data=None): Initializes the MultiFidelityDataManager object.
    - add_data(self, raw_fidelity_name, fidelity_index, x, y): Adds data to the data_dict.
    - get_data(self, fidelity_index, normal=True): Retrieves data from the data_dict.
    - get_data_by_name(self, raw_fidelity_name, normal=True): Retrieves data by fidelity name from the data_dict.
    - get_overlap_input_data(self, fidelity_index1, fidelity_index2, normal=False): Retrieves overlapping input data.
    - get_unique_input_data(self, fidelity_index1, fidelity_index2, normal=False): Retrieves unique input data.
    - get_nonsubset_fill_data(self, model, fidelity_index1, fidelity_index2): Generates filling data for non-subset data.
    - display_fidelity_data_info(self, fidelity_index): Displays information about the fidelity data.
    """
    
    def __init__(self, initial_data=None):
        """
        Initializes the MultiFidelityDataManager object.

        Parameters:
        - initial_data (list): A list of initial fidelity data.

        Returns:
        None
        """
        self.data_dict = {}
        self.normalizelayer = {}
        if initial_data is not None:
            for fidelity_data in initial_data:
                fidelity_name = fidelity_data['raw_fidelity_name']
                fidelity_index = fidelity_data['fidelity_indicator']
                x = fidelity_data['X']
                y = fidelity_data['Y']
                self.add_data(fidelity_name, fidelity_index, x, y)

    def add_data(self, raw_fidelity_name, fidelity_index, x, y):
        """
        Adds data to the data_dict.

        Parameters:
        - raw_fidelity_name (str): The name of the fidelity data.
        - fidelity_index (int): The index of the fidelity data.
        - x (torch.Tensor): The input data.
        - y (torch.Tensor): The output data.

        Returns:
        None
        """
        if raw_fidelity_name not in self.data_dict:
            self.data_dict[raw_fidelity_name] = {'fidelity_index': fidelity_index, 'X': x, 'Y': y}
        else:
            self.data_dict[raw_fidelity_name]['X'] = torch.cat([self.data_dict[raw_fidelity_name]['X'], x])
            self.data_dict[raw_fidelity_name]['Y'] = torch.cat([self.data_dict[raw_fidelity_name]['Y'], y])
        
        if fidelity_index not in self.normalizelayer and fidelity_index is not None:
            self.normalizelayer[fidelity_index] = Normalizer(x, y)

    def refresh_filling_data(self, raw_fidelity_name, fidelity_index, x, y):
        """
        Refreshes the filling data for a given raw fidelity name.

        Args:
            raw_fidelity_name (str): The name of the raw fidelity.
            fidelity_index (int): The fidelity index.
            x (list): The X data.
            y (list): The Y data.
        """
        if raw_fidelity_name not in self.data_dict:
            self.data_dict[raw_fidelity_name] = {'fidelity_index': fidelity_index, 'X': x, 'Y': y}
        else:
            self.data_dict[raw_fidelity_name]['X'] = x
            self.data_dict[raw_fidelity_name]['Y'] = y
    
    
    def get_data(self, fidelity_index, normal=False):
        """
        Retrieves data from the data_dict.

        Parameters:
        - fidelity_index (int): The index of the fidelity data.
        - normal (bool): Flag indicating whether to normalize the data.

        Returns:
        - x (torch.Tensor): The input data.
        - y (torch.Tensor): The output data.
        """
        for data in self.data_dict.values():
            if data['fidelity_index'] == fidelity_index:
                if normal and fidelity_index in self.normalizelayer:
                    return self.normalizelayer[fidelity_index].normalize(data['X'], data['Y'])
                else:
                    return data['X'], data['Y']
        
        warnings.warn("Can't find the data with the fidelity index: {}".format(fidelity_index))
        return None, None

    def get_data_by_name(self, raw_fidelity_name, normal = False):
        """
        Retrieves data by fidelity name from the data_dict.

        Parameters:
        - raw_fidelity_name (str): The name of the fidelity data.
        - normal (bool): Flag indicating whether to normalize the data.

        Returns:
        - x (torch.Tensor): The input data.
        - y (torch.Tensor): The output data.
        """
        if raw_fidelity_name in self.data_dict:
            if normal and self.data_dict[raw_fidelity_name]['fidelity_index'] in self.normalizelayer:
                return self.normalizelayer[self.data_dict[raw_fidelity_name]['fidelity_index']].normalize(self.data_dict[raw_fidelity_name]['X'], self.data_dict[raw_fidelity_name]['Y'])
            else:
                return self.data_dict[raw_fidelity_name]['X'], self.data_dict[raw_fidelity_name]['Y']
        else:
            warnings.warn("Can't find the data with the fidelity name: {}".format(raw_fidelity_name))
            return None, None

    def get_overlap_input_data(self, fidelity_index1, fidelity_index2, normal=False):
        """
        Retrieves overlapping input data.

        Parameters:
        - fidelity_index1 (int): The index of the first fidelity data.
        - fidelity_index2 (int): The index of the second fidelity data.
        - normal (bool): Flag indicating whether to normalize the data.

        Returns:
        - common_x1 (torch.Tensor): The overlapping input data for fidelity_index1.
        - y1 (torch.Tensor): The output data for fidelity_index1.
        - common_x2 (torch.Tensor): The overlapping input data for fidelity_index2.
        - y2 (torch.Tensor): The output data for fidelity_index2.
        """
        x1, y1 = self.get_data(fidelity_index1, normal=False)
        x2, y2 = self.get_data(fidelity_index2, normal=False)

        if x1 is not None and x2 is not None:
            mask_x1 = torch.all(x1.unsqueeze(dim=1) == x2.unsqueeze(dim=0), dim=-1)  # relative position mask
            mask_indices_x1 = torch.any(mask_x1, dim=-1)  # x1 index mask
            mask_x2 = torch.all(x2.unsqueeze(dim=1) == x1.unsqueeze(dim=0), dim=-1)
            mask_indices_x2 = torch.any(mask_x2, dim=-1)

            common_x1 = x1[mask_indices_x1]
            common_x2 = x2[mask_indices_x2]

            y1 = y1[mask_indices_x1]
            y2 = y2[mask_indices_x2]

            if normal and fidelity_index1 in self.normalizelayer and fidelity_index2 in self.normalizelayer:
                common_x1, y1 = self.normalizelayer[fidelity_index1].normalize(common_x1, y1)
                common_x2, y2 = self.normalizelayer[fidelity_index2].normalize(common_x2, y2)
            return common_x1, y1, common_x2, y2
        else:
            print("No overlap data found")
            return None, None, None, None

    def get_unique_input_data(self, fidelity_index1, fidelity_index2, normal=False):
        """
        Retrieves unique input data.

        Parameters:
        - fidelity_index1 (int): The index of the first fidelity data.
        - fidelity_index2 (int): The index of the second fidelity data.
        - normal (bool): Flag indicating whether to normalize the data.

        Returns:
        - unique_x1 (torch.Tensor): The unique input data for fidelity_index1.
        - y1 (torch.Tensor): The output data for fidelity_index1.
        - unique_x2 (torch.Tensor): The unique input data for fidelity_index2.
        - y2 (torch.Tensor): The output data for fidelity_index2.
        """
        x1, y1 = self.get_data(fidelity_index1, normal=False)
        x2, y2 = self.get_data(fidelity_index2, normal=False)

        if x1 is not None and x2 is not None:
            mask_x1 = torch.all(x1.unsqueeze(dim=1) == x2.unsqueeze(dim=0), dim=-1)  # relative position mask
            mask_indices_x1 = ~torch.any(mask_x1, dim=-1)  # x1 index mask
            mask_x2 = torch.all(x2.unsqueeze(dim=1) == x1.unsqueeze(dim=0), dim=-1)
            mask_indices_x2 = ~torch.any(mask_x2, dim=-1)

            unique_x1 = x1[mask_indices_x1]
            unique_x2 = x2[mask_indices_x2]

            y1 = y1[mask_indices_x1]
            y2 = y2[mask_indices_x2]

            if normal and fidelity_index1 in self.normalizelayer and fidelity_index2 in self.normalizelayer:
                unique_x1, y1 = self.normalizelayer[fidelity_index1].normalize(unique_x1, y1)
                unique_x2, y2 = self.normalizelayer[fidelity_index2].normalize(unique_x2, y2)
            return unique_x1, y1, unique_x2, y2
        else:
            print("No unique data found")
            return None, None, None, None

    def get_nonsubset_fill_data(self, model, fidelity_index1, fidelity_index2, normal = True):
        """
        Generates filling data for non-subset data.

        Parameters:
        - model: The fidelity fusion model.
        - fidelity_index1 (int): The index of the first fidelity data.
        - fidelity_index2 (int): The index of the second fidelity data.

        Returns:
        - x (torch.Tensor): The input data for filling.
        - y_low (list): The low-fidelity output data for filling.
        - y_high (list): The high-fidelity output data for filling.
        """
        # generate the filling data for the nonsubset data. 
        # this function requires a fidelity fusion model "model" following the formate of GAR, CIGAR, CAR, AR
        # If the user need to fill the data with different method, he can write his own function and replace this function. The key things it to use the first two lines of this function to get the subset data and nonsubset data.

        subset_x1, subset_y1, subset_x2, subset_y2 = self.get_overlap_input_data(fidelity_index1, fidelity_index2)
        unique_x1, unique_y1, unique_x2, unique_y2 = self.get_unique_input_data(fidelity_index1, fidelity_index2)

        if normal == True:
            _, subset_y1 = self.normalizelayer[fidelity_index1].normalize(subset_x1, subset_y1)
            subset_x2, subset_y2 = self.normalizelayer[fidelity_index2].normalize(subset_x2, subset_y2)
            unique_x2, unique_y2 = self.normalizelayer[fidelity_index2].normalize(unique_x2, unique_y2)

        ## full nonsubset: 
        if len(subset_x2) == 0:
            y_low_filling_mean, y_low_filling_var = model.forward(self, unique_x2, to_fidelity=fidelity_index1)
            if(y_low_filling_var.shape[0] != y_low_filling_var.shape[1]): ## because hogp only diagonal elements returned
                y_low_filling_var = torch.diag_embed(y_low_filling_var.squeeze())
            # y_high_var is zero because the outputs are observed
            y_high_var = torch.zeros((unique_y2.shape[0], unique_y2.shape[0]))
            return unique_x2, [y_low_filling_mean.reshape(-1, 1), y_low_filling_var], [unique_y2, y_high_var]
        ## full subset
        elif len(unique_x2) == 0:
            y_low_var = torch.zeros((subset_y1.shape[0], subset_y1.shape[0]))
            y_high_var = torch.zeros((subset_y2.shape[0], subset_y2.shape[0]))
            return subset_x2, [subset_y1, y_low_var], [subset_y2, y_high_var]
        else:
            y_low_filling_mean, y_low_filling_var = model.forward(self, unique_x2, to_fidelity=fidelity_index1)
            y_low_mean = torch.cat([subset_y1, y_low_filling_mean.reshape(-1, 1)], dim=0)
            if len(y_low_filling_mean.shape) == 0: ## if the y_low_filling_mean is a scalar
                y_low_filling_mean = y_low_filling_mean.reshape(1) #do it to make the y_low_filling_mean.shape[0] code work
            y_low_var = torch.zeros((subset_y1.shape[0] + y_low_filling_mean.shape[0], subset_y1.shape[0] + y_low_filling_mean.shape[0]))
            if(y_low_filling_var.shape[0] != y_low_filling_var.shape[1]): ## because hogp only diagonal elements returned
                y_low_filling_var = torch.diag_embed(y_low_filling_var.squeeze())
            y_low_var[-y_low_filling_var.shape[0]:, -y_low_filling_var.shape[1]:] = y_low_filling_var
            y_high_mean = torch.cat([subset_y2, unique_y2], dim=0)
            y_high_var = torch.zeros((subset_y2.shape[0] + unique_y2.shape[0], subset_y2.shape[0] + unique_y2.shape[0]))
            x = torch.cat([subset_x2, unique_x2], dim=0)
            return x, [y_low_mean, y_low_var], [y_high_mean, y_high_var]

    def display_fidelity_data_info(self, fidelity_index):
        """
        Display information about the fidelity data with the given fidelity index.

        Parameters:
        - fidelity_index (int): The index of the fidelity data to display.

        Returns:
        None
        """
        for raw_fidelity_name, data in self.data_dict.items():
            if data['fidelity_index'] == fidelity_index:
                print("<---------Fidelity data information:--------->")
                print("Fidelity index: {}".format(fidelity_index))
                print("Fidelity name: {}".format(raw_fidelity_name))
                print("data_num: {}".format(data['X'].shape[0]))
                print("X_shape: {}".format(data['X'].shape))
                print("Y_shape: {}".format(data['Y'].shape))
        else:
            print("No fidelity data found")

 
if __name__ == "__main__":

    initial_data = [
        {'raw_fidelity_name': 'IC_thermal','fidelity_indicator': 0 , 'X': torch.tensor([[1, 2], [2, 4], [3, 6], [4, 6], [7, 3]]), 'Y': torch.tensor([[5], [4], [3], [2], [1]])},
        {'raw_fidelity_name': '1','fidelity_indicator': 1 , 'X': torch.tensor([[1, 3], [2, 4], [7, 3], [6, 4], [7, 1]]), 'Y': torch.tensor([[7], [6], [5], [4], [3]])}
    ]

    fidelity_manager = MultiFidelityDataManager(initial_data)

    fidelity_manager.add_data(raw_fidelity_name = '2', fidelity_index = 2, x = torch.tensor([[2, 4], [3, 6]]), y = torch.tensor([[1.5], [2.5]]))

    x,y = fidelity_manager.get_data(-1)
    print("精度-1的数据:\nx:{}\ny:{}".format(x,y))

    common_x1, y1, common_x2, y2 = fidelity_manager.get_overlap_input_data(0, 1)
    print("0-1精度重合数据\nx1:{}\ny1:{}\nx2:{}\ny2:{}".format(common_x1, y1, common_x2, y2))

    unique_x1, y1, unique_x2, y2 = fidelity_manager.get_unique_input_data(0, 2)
    print("0-2精度不重合数据\nx1:{}\ny1:{}\nx2:{}\ny2:{}".format(unique_x1, y1, unique_x2, y2))

    x,y = fidelity_manager.get_data_by_name('1')
    print("IC_thermal的数据:\nx:{}\ny:{}".format(x,y))

    fidelity_manager.display_fidelity_data_info(0)

    pass