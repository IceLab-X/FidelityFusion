from copy import deepcopy
from utils.mlgp_log import mlgp_log

def smart_update(dict_base, dict_update):
    if dict_update is None:
        return dict_base

    dict_base = deepcopy(dict_base)
    dict_update = deepcopy(dict_update)

    mlgp_log.i("\nInput config as follow:")
    for _key, value in dict_update.items():
        mlgp_log.i(_key, ':', '{}'.format(value))
    mlgp_log.i("\n")

    _used_keys = []
    for _key, value in dict_update.items():
        if isinstance(value, dict):
            if _key in dict_base:
                dict_base[_key].update(value)
                _used_keys.append(_key)
            else:
                dict_base[_key] = value
                _used_keys.append(_key)
        else:
            continue
    for _key in _used_keys:
        dict_update.pop(_key)
    dict_base.update(dict_update)

    mlgp_log.i("\nInput config, after fullfill, meaning what actually feed to the model, as follow:")
    for _key, value in dict_base.items():
        mlgp_log.i(_key, ':', '{}'.format(value))
    mlgp_log.i("\n")

    return dict_base


def Dict_to_class(dict):
    class _class:
        def __init__(self, dict):
            for _key, _value in dict.items():
                if isinstance(_value, dict):
                    setattr(self, _key, Dict_to_class(_value))
                else:
                    setattr(self, _key, _value)
    return _class(dict)


def Class_to_dict(_class):
    tmp_vars = vars(_class)
    for _key, _value in tmp_vars.items():
        if isinstance(_value, _class):
            tmp_vars[_key] = Class_to_dict(_value)
    return tmp_vars