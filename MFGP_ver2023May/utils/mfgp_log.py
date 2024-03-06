

class MFGP_LOG(object):
    def __init__(self) -> None:
        self.log_level = 1

    @staticmethod
    def error(*args, **kargs):
        meg = ' '.join(args)
        print('\033[1;41m' + 'ERROR: ' + meg + '\033[0m', **kargs)

    @staticmethod
    def e(*args, **kargs):
        meg = ' '.join(args)
        print('\033[1;41m' + 'ERROR: ' + meg + '\033[0m', **kargs) 

    @staticmethod
    def info(*args, **kargs):
        meg = ' '.join(args)
        print('\033[1;40m' + 'INFO: ' + meg + '\033[0m', **kargs)

    @staticmethod
    def i(*args, **kargs):
        meg = ' '.join(args)
        print('\033[1;40m' + 'INFO: ' + meg + '\033[0m', **kargs)

    @staticmethod
    def warning(*args, **kargs):
        meg = ' '.join(args)
        print('\033[1;33m' + 'WARNING: ' + meg + '\033[0m',  **kargs)

    @staticmethod
    def w(*args, **kargs):
        meg = ' '.join(args)
        print('\033[1;33m' + 'WARNING: ' + meg + '\033[0m',  **kargs)

    @staticmethod
    def debug(*args, **kargs):
        meg = ' '.join(args)
        print('\033[1;45m' + 'DEBUG: ' + meg + '\033[0m', **kargs)

    @staticmethod
    def d(*args, **kargs):
        meg = ' '.join(args)
        print('\033[1;45m' + 'DEBUG: ' + meg + '\033[0m', **kargs)


if __name__ == '__main__':
    MFGP_LOG.i('This is info')
    MFGP_LOG.w('This is warning')
    MFGP_LOG.e('This is error')
    MFGP_LOG.d('This is debug')