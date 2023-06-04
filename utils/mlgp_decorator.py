from utils.mlgp_log import mlgp_log

def class_init_param_check(func):
    def inner(*args, **kwargs):
        _all_init_param = list(func.__init__.__code__.co_varnames)
        _default_values = list(func.__init__.__defaults__)
        _all_init_param = _all_init_param[1:] if _all_init_param[0]=='self' else _all_init_param[0]
        if len(args) + len(kwargs) > len(_all_init_param):
            mlgp_log.d('{} param check failed, len not match'.format(str(func)))
        elif len(args) + len(kwargs) <= len(_all_init_param):
            _default_values = [None for i in range(len(_all_init_param) - len(_default_values))] + _default_values
            _p_specify = {}
            _p_default = {}
            for i in range(len(args)):
                _p_specify[_all_init_param[i]] = args[i]
            for key, value in kwargs.items():
                _p_specify[key] = value
            for i, key in enumerate(_all_init_param):
                if key not in _p_specify:
                    _p_default[key] = _default_values[i]
            
            if len(_p_specify)>0 and True: # set log level here
                mesg = '{} define using specify param as:'.format(str(func))
                for key, value in _p_specify.items():
                    mesg = mesg + ' ' + key + ' = ' + str(value) + ';'
                mlgp_log.d(mesg)
            
            if len(_p_default)>0 and True: # set log level here
                mesg = '{} define using default param as:'.format(str(func))
                for key, value in _p_default.items():
                    mesg = mesg + ' ' + key + ' = ' + str(value) + ';'
                mlgp_log.d(mesg)
        return func(*args, **kwargs)
    return inner