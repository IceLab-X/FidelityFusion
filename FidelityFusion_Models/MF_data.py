import torch

class MultiFidelityDataManager:
    """
    A class for managing multi-fidelity data.

    Attributes:
        data_dict (dict): A dictionary to store the data for each fidelity index.

    Methods:
        __init__(self, initial_data=None): Initializes the MultiFidelityDataManager object.
        add_data(self, fidelity_index, x, y): Adds data for a specific fidelity index.
        get_data(self, fidelity_index): Retrieves the data for a specific fidelity index.
        get_overlap_input_data(self, fidelity_index1, fidelity_index2): Retrieves the overlapping input data between two fidelity indices.
        get_unique_input_data(self, fidelity_index1, fidelity_index2): Retrieves the unique input data between two fidelity indices.
    """

    def __init__(self, initial_data=None):
        """
        Initializes the MultiFidelityDataManager object.

        Args:
            initial_data (list, optional): A list of dictionaries containing initial data for each fidelity index.
        """
        self.data_dict = {}
        if initial_data is not None:
            for fidelity_data in initial_data:
                fidelity_index = fidelity_data['fidelity_index']
                x = fidelity_data['X']
                y = fidelity_data['Y']
                self.add_data(fidelity_index, x, y)

    def add_data(self, fidelity_index, x, y):
        """
        Adds data for a specific fidelity index.

        Args:
            fidelity_index (int): The fidelity index.
            x (torch.Tensor): The input data.
            y (torch.Tensor): The output data.
        """
        if fidelity_index not in self.data_dict:
            self.data_dict[fidelity_index] = {'X': x, 'Y': y}
        else:
            self.data_dict[fidelity_index]['X'] = torch.cat([self.data_dict[fidelity_index]['X'], x])
            self.data_dict[fidelity_index]['Y'] = torch.cat([self.data_dict[fidelity_index]['Y'], y])

    def get_data(self, fidelity_index):
        """
        Retrieves the data for a specific fidelity index.

        Args:
            fidelity_index (int): The fidelity index.

        Returns:
            tuple: A tuple containing the input data (X) and output data (Y) for the specified fidelity index.
                   Returns None if no data is found for the fidelity index.
        """
        if fidelity_index in self.data_dict:
            return self.data_dict[fidelity_index]['X'], self.data_dict[fidelity_index]['Y']
        else:
            return None

    def get_overlap_input_data(self, fidelity_index1, fidelity_index2):
        """
        Retrieves the overlapping input data between two fidelity indices.

        Args:
            fidelity_index1 (int): The first fidelity index.
            fidelity_index2 (int): The second fidelity index.

        Returns:
            tuple: A tuple containing the common input data (common_x1, common_x2) and the corresponding output data (y1, y2).
                   Returns None if no overlap data is found.
        """
        data1 = self.get_data(fidelity_index1)
        data2 = self.get_data(fidelity_index2)

        if data1 is not None and data2 is not None:
            common_indices_x1 = torch.all(data1[0][:, None, :] == data2[0][None, :, :], dim=-1).any(dim=-1)
            common_indices_x2 = torch.all(data2[0][:, None, :] == data1[0][None, :, :], dim=-1).any(dim=-1)

            common_x1 = data1[0][common_indices_x1]
            common_x2 = data2[0][common_indices_x2]

            # sorted_indices_x1 = torch.argsort(common_x1, dim=0)
            # sorted_indices_x2 = torch.argsort(common_x2, dim=0)

            # common_x1 = common_x1[sorted_indices_x1]
            # common_x2 = common_x2[sorted_indices_x2]

            # y1 = data1[1][common_indices_x1][sorted_indices_x1]
            # y2 = data2[1][common_indices_x2][sorted_indices_x2]
            y1 = data1[1][common_indices_x1]
            y2 = data2[1][common_indices_x2]

            return common_x1, y1, common_x2, y2
        else:
            print("No overlap data found")
            return None

    def get_unique_input_data(self, fidelity_index1, fidelity_index2):
        """
        Retrieves the unique input data between two fidelity indices.

        Args:
            fidelity_index1 (int): The first fidelity index.
            fidelity_index2 (int): The second fidelity index.

        Returns:
            tuple: A tuple containing the unique input data (unique_x1, unique_x2) and the corresponding output data (y1, y2).
                   Returns None if no unique data is found.
        """
        data1 = self.get_data(fidelity_index1)
        data2 = self.get_data(fidelity_index2)

        if data1 is not None and data2 is not None:
            unique_indices1 = ~torch.all(data1[0][:, None, :] == data2[0][None, :, :], dim=-1).any(dim=-1)
            unique_indices2 = ~torch.all(data2[0][:, None, :] == data1[0][None, :, :], dim=-1).any(dim=-1)

            unique_x1 = data1[0][unique_indices1]
            unique_x2 = data2[0][unique_indices2]

            y1 = data1[1][unique_indices1]
            y2 = data2[1][unique_indices2]

            return unique_x1, y1, unique_x2, y2
        else:
            print("No unique data found")
            return None

if __name__ == "__main__":

    initial_data = [
        {'fidelity_index': '0', 'X': torch.tensor([[1, 2], [2, 4], [3, 6], [4, 6], [7, 3]]), 'Y': torch.tensor([[5], [4], [3], [2], [1]])},
        {'fidelity_index': '1', 'X': torch.tensor([[1, 3], [2, 4], [7, 3], [6, 4], [7, 1]]), 'Y': torch.tensor([[7], [6], [5], [4], [3]])}
    ]

    fidelity_manager = MultiFidelityDataManager(initial_data)

    fidelity_manager.add_data('2', torch.tensor([[2, 4,1,1], [3, 6,1,1]]), torch.tensor([[1.5], [2.5]]))

    x,y = fidelity_manager.get_data('2')
    print("精度2的数据:\nx:{}\ny:{}".format(x,y))

    overlap_data = fidelity_manager.get_overlap_input_data('0', '1')
    print("0-1精度重合的数据:", overlap_data)

    unique_data = fidelity_manager.get_unique_input_data('0', '2')
    print("0-2精度不重合的数据:", unique_data)
