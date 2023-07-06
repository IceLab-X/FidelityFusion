import os
import sys

import yaml

from mffusion.utils.mlgp_log import mlgp_log
from copy import deepcopy


def _dict_to_str(tar_dict):
    def _parse_dict(_d, _srt_list, _depth=0):
        _blank = '  '
        for _key, _value in _d.items():
            if isinstance(_value, dict):
                _str.append(_blank*_depth + _key + ':\n')
                _parse_dict(_value, _srt_list, _depth+1)
            else:
                _srt_list.append(_blank*_depth + _key + ': ' + str(_value) + '\n')

    if not isinstance(tar_dict, dict):
        mlgp_log.e('{} is not a dict'.format(tar_dict))

    _str = []
    _parse_dict(tar_dict, _str)
    return _str


class MLGP_recorder:
    def __init__(self, save_path, append_info=None, overlap=False) -> None:
        self.save_path = save_path
        self._f = None
        self._register_state = False
        self._register_len = None

        self._key = None
        self._record_list = []

        # create folder
        _folder = os.path.dirname(save_path)
        if not os.path.exists(_folder):
            os.makedirs(_folder)

        if os.path.exists(save_path) and overlap is False:
            mlgp_log.e("[{}] is already exists, create failed, set overlap=True to avoid this check".format(save_path))
            raise RuntimeError()

        if overlap and os.path.exists(save_path):
            self._f = open(save_path, 'a')
            self._f.write('\n\n')
        else:
            self._f = open(save_path, 'w')
        
        self._f.write("@MLGP_recorder@\n")
        if append_info is not None:
            self._write_append_info(append_info)
        self._f.flush()


    def _write_append_info(self, info):
        self._f.write('@append_info@\n')
        if isinstance(info, dict):
            _str = _dict_to_str(info)
            for _s in _str:
                self._f.write(_s)
        elif isinstance(info, list):
            for _s in info:
                self._f.write(str(_s))
        else:
            self._f.write(str(info))
        self._f.write('@append_info@\n')

    def register(self, key_list, re_register=False):
        if self._register_state == True and \
            re_register is False:
            mlgp_log.e("recorder has been register, double register is not allow, unless set overlap=True")
            raise RuntimeError()

        self._key = deepcopy(key_list)
        self._register_len = len(key_list)
        self._register_state = True
        self._f.write('@record_result@\n')
        self.record(key_list)


    def record(self, _single_record, check_len=True):
        if not isinstance(_single_record, list) and not isinstance(_single_record, dict):
            mlgp_log.e("MLGP_recorder.record only accept input as dict/list")

        if len(_single_record) != self._register_len and check_len is True:
            mlgp_log.w("record failed, {} len is not match key list: {}".format(len(_single_record), self._key))

        if isinstance(_single_record, dict):
            _backup = deepcopy(_single_record)
            _single_record = [None] * len(_backup)
            for _k, _v in _backup.items():
                _single_record[self._key.index(_k)] = _v

        self._record_list.append(deepcopy(_single_record))

        _single_record = [str(_sr) for _sr in _single_record]
        _single_record[-1] = _single_record[-1] + '\n'

        self._f.write(','.join(_single_record))
        self._f.flush()

    def to_csv(self, seed='default', csv_path=None):
        if csv_path is None:
            csv_path = self.save_path.replace('.txt', '_' + str(seed) + '.csv')

        import csv
        with open(csv_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerows(self._record_list)
        return


class MLGP_record_parser:
    def __init__(self, path) -> None:
        self._f = open(path, 'rb')
        self._f.seek(0, 2)
        self.file_len = self._f.tell()
        self._f.seek(0, 0)

        self.start_pattern = '@MLGP_recorder@'
        self.ex_pattern = ['@append_info@', '@record_result@', '@exception_catch_log@']
        assert len(self.ex_pattern) == len(set(self.ex_pattern)), "repeat pattern is not allow"

        self._d = '\r\n'

        # general matching, working for every case
        self._pattrn_list = []
        self._match_pattern()

        # only work for mlgp
        self.reformat_list = []
        self._reformat_pattern()

    def _match_pattern(self):
        _len_bf = len(self._pattrn_list)
        while self._f.tell() < self.file_len:
            _l = self._f.readline().decode('utf-8')
            if _l.rstrip(self._d) == self.start_pattern:
                ex_pattern_dict = self._match_ex_pattern()
                self._pattrn_list.append(deepcopy(ex_pattern_dict))
        mlgp_log.i('Got {} valid data after match pattern'.format(len(self._pattrn_list) - _len_bf))
        return

    def _match_ex_pattern(self):
        _ex_pattern_dict = {}
        _single_list = []
        _pattern_now = None
        while self._f.tell() < self.file_len:
            _l = self._f.readline().decode('utf-8')

            # meet start parttern, jumpout
            if _l.rstrip(self._d) == self.start_pattern:
                self._f.seek(self._f.tell()- len(self.start_pattern+'\r\n'))
                break

            if _l.rstrip(self._d) in self.ex_pattern and \
                _l.rstrip(self._d) != _pattern_now:
                if _pattern_now is not None:
                    _ex_pattern_dict[_pattern_now] = deepcopy(_single_list)
                _pattern_now = _l.rstrip(self._d)
                _single_list = []
            elif _pattern_now is not None and \
                _l.strip(self._d) != _pattern_now:
                _single_list.append(_l)

        if _pattern_now not in _ex_pattern_dict:
            _ex_pattern_dict[_pattern_now] = deepcopy(_single_list)

        return _ex_pattern_dict

    def _match_record(self):
        _record_list = []
        while True:
            _l = self._f.readline()
            if _l == '' or \
               _l == '\n':
                break
            elif _l == self.start_pattern + '\n':
                self._f.seek(self._f.tell()-1)
                break
            else:
                _record_list.append(_l)
        return _record_list

    def _reformat_pattern(self):
        for i in range(len(self._pattrn_list)):
            _rl = {}
            _rl[self.ex_pattern[0]] = self._str_list_to_yaml(self._pattrn_list[i][self.ex_pattern[0]])
            _rl[self.ex_pattern[0] + '_ref'] = self._resolve_sub_dict(_rl[self.ex_pattern[0]])
            _rl[self.ex_pattern[1]] =(self._record_as_num(self._pattrn_list[i][self.ex_pattern[1]]))
            self.reformat_list.append(deepcopy(_rl))
        return

    def _str_list_to_yaml(self, _list):
        with open('./tmp.yaml', 'w') as f:
            for _l in _list:
                f.write(_l)
        with open('./tmp.yaml','r') as f:
            yaml_data = yaml.load(f, Loader=yaml.FullLoader)
        # os.system('rm ./tmp.yaml')
        return yaml_data

    def _resolve_sub_dict(self, _dict):
        new_dict = {}
        def _resolve_dict(_dict, father, _nd):
            father = father + '.' if father != '' else father
            for key, item in _dict.items():
                if isinstance(item, dict):
                    _resolve_dict(item, father + key, _nd)
                else:
                    _nd[father + key] = item
        _resolve_dict(_dict, '', new_dict)
        return new_dict

    def _record_as_num(self, _list):
        _list = deepcopy(_list)
        while _list[-1] == self._d:
            _list.pop()

        for i, _l in enumerate(_list):
            _list[i] = _l.rstrip(self._d)

        for i, _l in enumerate(_list):
            _list[i] = _list[i].split(',')
            if i>0:
                if not isinstance(_list[i], list) or \
                    _list[i] == ['']:
                    continue
                _list[i] = [float(_v) for _v in _list[i]]

        # hack for mac. wait for optimize
        _new_list = []
        for _l in _list:
            if _l != ['']:
                _new_list.append(_l)

        return _new_list

    def get_data(self):
        return self.reformat_list
    
    def record_to_csv(self, csv_path):
        import csv
        with open(csv_path, 'w', newline='') as f:
            writer = csv.writer(f)

            for _l in self.reformat_list:
                record_result = _l['@record_result@']
                # for line in record_result:
                writer.writerow([_l['@append_info@']])
                writer.writerows(record_result)
                writer.writerow(" ")
        return


if __name__ == '__main__':
    import datetime
    txt_path = './record_test.txt'
    append_info = {'Function': 'A', 
                    'Purpose': 'Test',
                    'time': {
                    'now': datetime.datetime.today(),
                    'weekdata': datetime.datetime.today().isoweekday()}
                    }
    rc = MLGP_recorder(txt_path, append_info, overlap=True)
    rc.register(['epoch','result'])
    rc.record([0, 0.5])
    rc.record({'epoch': 1, 'result': 0.6})
    rc.record({'result': 0.7, 'epoch': 2})

    paser = MLGP_record_parser('./record_test.txt')
    print(paser.get_data())
    paser.record_to_csv('./record_test.csv')