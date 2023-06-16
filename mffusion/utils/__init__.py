from mffusion.utils.eigen import eigen_pairs
from mffusion.utils.normalizer import Normalizer
from mffusion.utils.performance_evaluator import performance_evaluator, high_level_evaluator
from mffusion.utils.ResPCA import listPCA, resPCA_mf
from mffusion.utils.dict_tools import smart_update
from mffusion.utils.mlgp_log import mlgp_log
from mffusion.utils.mlgp_hook import register_nan_hook

JITTER = 1e-6
EPS = 1e-10
PI = 3.1415