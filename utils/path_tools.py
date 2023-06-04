import os
from utils import mlgp_log


default_name = 'exp'


def smart_sep(dir_path):
    if os.sep == "/":
        dir_path = dir_path.replace("\\", "/")
    elif os.sep == "\\":
        dir_path = dir_path.replace("/", "\\")
    return dir_path


def deep_mkdir(dir_path):
    dir_path = smart_sep(dir_path)
    elements = dir_path.split(os.sep)
    for i in range(len(elements)):
        _path = os.sep.join(elements[:i+1])
        if not os.path.exists(_path):
            os.mkdir(_path)


def check_dir(dir_path):
    if not os.path.isdir(dir_path):
        mlgp_log.e("{} is not folder".format(dir_path))


def get_available_name(dir_path, file_format='txt'):
    check_dir(dir_path)
    index = 0
    _path = dir_path
    while os.path.exists(_path):
        index += 1
        _path = os.path.join(dir_path, default_name + '_{}.{}'.format(index,file_format))
    return _path


def get_last_name(dir_path, file_format='txt'):
    check_dir(dir_path)
    index = 0
    _path = dir_path
    while os.path.exists(_path):
        index += 1
        _path = os.path.join(dir_path, default_name + '_{}.{}'.format(index,file_format))
    _path = os.path.join(dir_path, default_name + '_{}.{}'.format(index-1,file_format))
    return _path