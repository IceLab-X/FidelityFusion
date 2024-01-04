'''
version 1.0 : 2023/12/28 finish init and add_data, get_data, get_overlap_input_data, get_unique_input_data
version 1.1 : 2024/1/4 finish display_fidelity_data and optimized code
'''
import torch

class MultiFidelityDataManager:

    def __init__(self, initial_data=None):
        
        self.data_dict = {}
        if initial_data is not None:
            for fidelity_data in initial_data:
                fidelity_index = fidelity_data['fidelity_indicator']
                fidelity_name = fidelity_data['raw_fidelity_name']
                x = fidelity_data['X']
                y = fidelity_data['Y']
                self.add_data(fidelity_index, fidelity_name, x, y)

    def add_data(self, fidelity_index, raw_fidelity_name, x, y):
        
        if fidelity_index not in self.data_dict:
            self.data_dict[fidelity_index] = {'fidelity_name':raw_fidelity_name,'X': x, 'Y': y}
        else:
            self.data_dict[fidelity_index]['X'] = torch.cat([self.data_dict[fidelity_index]['X'], x])
            self.data_dict[fidelity_index]['Y'] = torch.cat([self.data_dict[fidelity_index]['Y'], y])

    def get_data(self, fidelity_index):
        
        if fidelity_index in self.data_dict:
            return self.data_dict[fidelity_index]['X'], self.data_dict[fidelity_index]['Y']
        else:
            return None , None
        
    def get_data_by_name(self, raw_fidelity_name):

        for data in self.data_dict.values():
            if data['fidelity_name'] == raw_fidelity_name:
                return data['X'], data['Y']
        return None, None

    def get_overlap_input_data(self, fidelity_index1, fidelity_index2):
        
        x1,y1 = self.get_data(fidelity_index1)
        x2,y2 = self.get_data(fidelity_index2)

        if x1 is not None and x2 is not None:
            
            mask_x1 = torch.all(x1.unsqueeze(dim=1) == x2.unsqueeze(dim=0), dim=-1) # relative position mask
            mask_indices_x1 = torch.any(mask_x1, dim=-1) # x1 index mask
            mask_x2 = torch.all(x2.unsqueeze(dim=1) == x1.unsqueeze(dim=0), dim=-1)
            mask_indices_x2 = torch.any(mask_x2, dim=-1)
    
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
            
            mask_x1 = torch.all(x1.unsqueeze(dim=1) == x2.unsqueeze(dim=0), dim=-1) # relative position mask
            mask_indices_x1 = ~torch.any(mask_x1, dim=-1) # x1 index mask
            mask_x2 = torch.all(x2.unsqueeze(dim=1) == x1.unsqueeze(dim=0), dim=-1)
            mask_indices_x2 = ~torch.any(mask_x2, dim=-1)

            unique_x1 = x1[mask_indices_x1]
            unique_x2 = x2[mask_indices_x2]

            y1 = y1[mask_indices_x1]
            y2 = y2[mask_indices_x2]

            return unique_x1, y1, unique_x2, y2
        else:
            print("No unique data found")
            return None , None , None , None
    
    def display_fidelity_data_info(self, fidelity_index):
        
        if fidelity_index in self.data_dict:
            print("<---------Fidelity data informaton:--------->")
            print("Fidelity index: {}".format(fidelity_index))
            print("Fidelity name: {}".format(self.data_dict[fidelity_index]['fidelity_name']))
            print("data_num: {}".format(self.data_dict[fidelity_index]['X'].shape[0]))
            print("X_shape: {}".format(self.data_dict[fidelity_index]['X'].shape))
            print("Y_shape: {}".format(self.data_dict[fidelity_index]['Y'].shape))
        else:
            print("No fidelity data found")

##display 
if __name__ == "__main__":

    initial_data = [
        {'fidelity_indicator': 0 ,'raw_fidelity_name': 'IC_thermal', 'X': torch.tensor([[1, 2], [2, 4], [3, 6], [4, 6], [7, 3]]), 'Y': torch.tensor([[5], [4], [3], [2], [1]])},
        {'fidelity_indicator': 1 ,'raw_fidelity_name': '1','X': torch.tensor([[1, 3], [2, 4], [7, 3], [6, 4], [7, 1]]), 'Y': torch.tensor([[7], [6], [5], [4], [3]])}
    ]

    fidelity_manager = MultiFidelityDataManager(initial_data)

    fidelity_manager.add_data(fidelity_index=-1,raw_fidelity_name='-1',x=torch.tensor([[2, 4], [3, 6]]),y=torch.tensor([[1.5], [2.5]]))

    x,y = fidelity_manager.get_data(-1)
    print("精度-1的数据:\nx:{}\ny:{}".format(x,y))

    common_x1, y1, common_x2, y2 = fidelity_manager.get_overlap_input_data(0, 2)
    print("0-1精度重合数据\nx1:{}\ny1:{}\nx2:{}\ny2:{}".format(common_x1, y1, common_x2, y2))

    unique_x1, y1, unique_x2, y2 = fidelity_manager.get_unique_input_data(0, 2)
    print("0-2精度不重合数据\nx1:{}\ny1:{}\nx2:{}\ny2:{}".format(unique_x1, y1, unique_x2, y2))

    x,y = fidelity_manager.get_data_by_name('1')
    print("IC_thermal的数据:\nx:{}\ny:{}".format(x,y))

    fidelity_manager.display_fidelity_data_info(0)

    pass