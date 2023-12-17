import torch

def update_dict_with_default(default_dict, update_dict):
    if update_dict is None:
        return default_dict

    for key in default_dict.keys():
        if key not in update_dict.keys():
            update_dict[key] = default_dict[key]
    return update_dict