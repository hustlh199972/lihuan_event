import os
import logging
from pathlib import Path
from functools import reduce, partial
from operator import getitem
from datetime import datetime
from logger import setup_logging
from utils.util import read_json, write_json


class ConfigParser:
    def __init__(self, config, resume=None, modification=None, run_id=None):
        # load config file and apply modification
        self._config = _update_config(config, modification)
        self.resume = resume

        # set save_dir where trained model and log will be saved.
        save_dir = Path(self.config['trainer']['save_dir'])

        exper_name = self.config['name']
        if run_id is None: # use timestamp as default run-id
            run_id = datetime.now().strftime(r'%m%d_%H%M%S')#返回当前的日期和时间,格式化为指定的字符串格式
        self._save_dir = save_dir / 'models' / exper_name / run_id
        self._log_dir = save_dir / 'log' / exper_name / run_id

        # make directory for saving checkpoints and log.
        exist_ok = run_id == ''
        self.save_dir.mkdir(parents=True, exist_ok=exist_ok)    #已存在，不需要；不存在该文件夹，创建新的
        self.log_dir.mkdir(parents=True, exist_ok=exist_ok)

        # save updated config file to the checkpoint dir
        write_json(self.config, self.save_dir / 'config.json')  #把config写入save_dir / 'config.json‘

        # configure logging module
        setup_logging(self.log_dir)
        self.log_levels = {
            0: logging.WARNING,
            1: logging.INFO,
            2: logging.DEBUG
        }

    @classmethod    #不用实例化，可以直接调用
    def from_args(cls, args, options=''):
        """
        Initialize this class from some cli arguments. Used in train, test.
        """
        for opt in options:
            args.add_argument(*opt.flags, default=None, type=opt.type)
        if not isinstance(args, tuple): #如果args不是元组执行：
            args = args.parse_args()    ## 解析参数

        if args.device is not None:
            os.environ["CUDA_VISIBLE_DEVICES"] = args.device
        if args.resume is not None: #从检查点恢复：
            resume = Path(args.resume)
            cfg_fname = resume.parent / 'config.json'   #resume的父目录
        else:       #检查是否配置了Configuration文件
            msg_no_cfg = "Configuration file need to be specified. Add '-c config.json', for example."
            assert args.config is not None, msg_no_cfg
            resume = None
            cfg_fname = Path(args.config)

        config = read_json(cfg_fname)   #读取 JSON 文件，并返回一个有序字典
        if args.config and resume:
            # update new config for fine-tuning
            config.update(read_json(args.config))

        # parse custom cli options into dictionary      #获取options的target:args对应的flag(去掉--)
        modification = {opt.target : getattr(args, _get_opt_name(opt.flags)) for opt in options}
        return cls(config, resume, modification)    #cls指向类本身，实例化创建一个 ConfigParser 类的对象

    def init_obj(self, name, module, *args, **kwargs):
        module_name = self[name]['type']
        module_args = dict(self[name]['args'])
        assert all([k not in module_args for k in kwargs]), 'Overwriting kwargs given in config file is not allowed'
        module_args.update(kwargs)  #将kwargs添加到args字典中
        return getattr(module, module_name)(*args, **module_args)   #*args 用于接收任意数量的位置参数，这些参数会被封装为一个元组，参数个数必须一致
                                                        #**module_args封装为一个字典，**选择性地将字典中的参数传递给函数，参数个数可以不一致
    def init_ftn(self, name, module, *args, **kwargs):
        """
        Finds a function handle with the name given as 'type' in config, and returns the
        function with given arguments fixed with functools.partial.

        `function = config.init_ftn('name', module, a, b=1)`
        is equivalent to
        `function = lambda *args, **kwargs: module.name(a, *args, b=1, **kwargs)`.
        """
        module_name = self[name]['type']
        module_args = dict(self[name]['args'])
        assert all([k not in module_args for k in kwargs]), 'Overwriting kwargs given in config file is not allowed'
        module_args.update(kwargs)
        return partial(getattr(module, module_name), *args, **module_args)

    def __getitem__(self, name):
        """Access items like ordinary dict."""
        return self.config[name]

    def get_logger(self, name, verbosity=2):
        msg_verbosity = 'verbosity option {} is invalid. Valid options are {}.'.format(verbosity, self.log_levels.keys())
        assert verbosity in self.log_levels, msg_verbosity
        logger = logging.getLogger(name)
        logger.setLevel(self.log_levels[verbosity])
        return logger

    # setting read-only attributes
    @property   #调用时无需加括号
    def config(self):
        return self._config

    @property
    def save_dir(self):
        return self._save_dir

    @property
    def log_dir(self):
        return self._log_dir

# helper functions to update config dict with custom cli options
def _update_config(config, modification):
    if modification is None:
        return config

    for k, v in modification.items():
        if v is not None:
            _set_by_path(config, k, v)
    return config

def _get_opt_name(flags):
    for flg in flags:
        if flg.startswith('--'):
            return flg.replace('--', '')
    return flags[0].replace('--', '')

def _set_by_path(tree, keys, value):
    """Set a value in a nested object in tree by sequence of keys."""
    keys = keys.split(';')
    _get_by_path(tree, keys[:-1])[keys[-1]] = value

def _get_by_path(tree, keys):
    """Access a nested object in tree by sequence of keys."""
    return reduce(getitem, keys, tree)
