'''
version 1.0 : 2023/12/28 finish init and add_data, get_data, get_overlap_input_data, get_unique_input_data
version 1.1 : 2024/1/4 finish display_fidelity_data and optimized code
version 1.2 : 2024/1/6 add get_data_by_name and get_nonsubset_data
'''
import torch

class MultiFidelityDataManager:

    def __init__(self, initial_data=None):
        
        self.data_dict = {}
        if initial_data is not None:
            for fidelity_data in initial_data:
                fidelity_name = fidelity_data['raw_fidelity_name']
                fidelity_index = fidelity_data['fidelity_indicator']
                x = fidelity_data['X']
                y = fidelity_data['Y']
                self.add_data(fidelity_name, fidelity_index, x, y)

    def add_data(self, raw_fidelity_name, fidelity_index, x, y):
        
        if raw_fidelity_name not in self.data_dict:
            self.data_dict[raw_fidelity_name] = {'fidelity_index':fidelity_index, 'X': x, 'Y': y}
        else:
            self.data_dict[raw_fidelity_name]['X'] = torch.cat([self.data_dict[raw_fidelity_name]['X'], x])
            self.data_dict[raw_fidelity_name]['Y'] = torch.cat([self.data_dict[raw_fidelity_name]['Y'], y])

    def get_data(self, fidelity_index):
        
        for data in self.data_dict.values():
            if data['fidelity_index'] == fidelity_index:
                return data['X'], data['Y']
        return None, None

        
    def get_data_by_name(self, raw_fidelity_name):

        if raw_fidelity_name in self.data_dict:
            return self.data_dict[raw_fidelity_name]['X'], self.data_dict[raw_fidelity_name]['Y']
        else:
            return None , None


    def get_overlap_input_data(self, fidelity_index1, fidelity_index2):
        
        x1,y1 = self.get_data(fidelity_index1)
        x2,y2 = self.get_data(fidelity_index2)

        if x1 is not None and x2 is not None:
            
            mask_x1 = torch.all(x1.unsqueeze(dim = 1) == x2.unsqueeze(dim = 0), dim=-1) # relative position mask
            mask_indices_x1 = torch.any(mask_x1, dim=-1) # x1 index mask
            mask_x2 = torch.all(x2.unsqueeze(dim = 1) == x1.unsqueeze(dim = 0), dim=-1)
            mask_indices_x2 = torch.any(mask_x2, dim = -1)
    
            common_x1 = x1[mask_indices_x1]
            common_x2 = x2[mask_indices_x2]

            y1 = y1[mask_indices_x1]
            y2 = y2[mask_indices_x2]

            return common_x1, y1, common_x2, y2
        else:
            print("No overlap data found")
            return None, None, None, None

    def get_unique_input_data(self, fidelity_index1, fidelity_index2):
        
        x1,y1 = self.get_data(fidelity_index1)
        x2,y2 = self.get_data(fidelity_index2)

        if x1 is not None and x2 is not None:
            
            mask_x1 = torch.all(x1.unsqueeze(dim = 1) == x2.unsqueeze(dim = 0), dim = -1) # relative position mask
            mask_indices_x1 = ~torch.any(mask_x1, dim=-1) # x1 index mask
            mask_x2 = torch.all(x2.unsqueeze(dim = 1) == x1.unsqueeze(dim = 0), dim = -1)
            mask_indices_x2 = ~torch.any(mask_x2, dim=-1)

            unique_x1 = x1[mask_indices_x1]
            unique_x2 = x2[mask_indices_x2]

            y1 = y1[mask_indices_x1]
            y2 = y2[mask_indices_x2]

            return unique_x1, y1, unique_x2, y2
        else:
            print("No unique data found")
            return None , None , None , None
    
    def get_nonsubset_data(self, GPmodel, fidelity_index1, fidelity_index2):
            
        subset_x1, subset_y1, subset_x2, subset_y2 = self.get_overlap_input_data(fidelity_index1, fidelity_index2)
        unique_x1, unique_y1, unique_x2, unique_y2 = self.get_unique_input_data(fidelity_index1, fidelity_index2)

        ## full nonsubset
        if len(subset_x2) == 0: 
            y_low_filling_mean,y_low_filling_var = GPmodel.forward(unique_x2,to_fidelity = fidelity_index2)
            y_high_var = torch.zeros((unique_y2.shape[0], unique_y2.shape[0]))
            return unique_x2 , [y_low_filling_mean , y_low_filling_var] , [unique_y2 , y_high_var]
        ## full subset
        elif len(unique_x2) == 0: 
            y_low_var = torch.zeros((subset_y1.shape[0], subset_y1.shape[0]))
            y_high_var = torch.zeros((subset_y2.shape[0], subset_y2.shape[0]))
            return subset_x2 , [subset_y1 , y_low_var], [subset_y2 , y_high_var]
        else: 
            y_low_filling_mean, y_low_filling_var = GPmodel.forward(self, unique_x2, to_fidelity=fidelity_index2)
            y_low_mean = torch.cat([subset_y1, y_low_filling_mean.reshape(-1,1)], dim = 0)
            y_low_var = torch.zeros((subset_y1.shape[0] + y_low_filling_mean.shape[0], subset_y1.shape[0] + y_low_filling_mean.shape[0]))
            y_low_var[-y_low_filling_var.shape[0]:, -y_low_filling_var.shape[1]:] = y_low_filling_var
            y_high_mean = torch.cat([subset_y2, unique_y2], dim = 0)
            y_high_var = torch.zeros((subset_y2.shape[0] + unique_y2.shape[0], subset_y2.shape[0] + unique_y2.shape[0]))
            x = torch.cat([subset_x2,unique_x2], dim = 0)
            return x , [y_low_mean ,y_low_var] , [y_high_mean , y_high_var]
            
    def display_fidelity_data_info(self, fidelity_index):

        for raw_fidelity_name , data in self.data_dict.items():
            if data['fidelity_index'] == fidelity_index:
                print("<---------Fidelity data informaton:--------->")
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