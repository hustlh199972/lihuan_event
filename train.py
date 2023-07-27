import argparse
import collections
import torch
import numpy as np
import data_loader.data_loaders as module_data
import model.loss as module_loss
import model.model as module_arch
from inference import *
from parse_config import ConfigParser
from trainer import Trainer
'''
python train.py --config config/reconstruction.json
'''
# fix random seeds for reproducibility
SEED = 123
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True# 设置 PyTorch 在使用 GPU 加速时的随机性
torch.backends.cudnn.benchmark = False
np.random.seed(SEED)

def load_model(args, checkpoint=None, config=None):
    """
    negative voxel indicates a model trained on negative voxels -.-
    """
    resume = checkpoint is not None
    if resume:
        config = checkpoint['config']
        state_dict = checkpoint['state_dict']
    try:
        model_info['num_bins'] = config['arch']['args']['unet_kwargs']['num_bins']
    except KeyError:
        model_info['num_bins'] = config['arch']['args']['num_bins']
    logger = config.get_logger('test')  #回一个日志记录器，该记录器与给定的名称('test')相关联

    if args.legacy:
        config['arch']['type'] += '_legacy'
    # build model architecture
    model = config.init_obj('arch', module_arch)    #从module_arch返回arch:type的模型
    logger.info(model)

    if config['n_gpu'] > 1:
        model = torch.nn.DataParallel(model)    #多个GPU
    if resume:
        model.load_state_dict(state_dict)   #加载权重

    # prepare model for testing
    model = model.to(device)
    model.eval()
    if args.color:
        model = ColorNet(model)
        print('Loaded ColorNet')
    return model

def main(config):   #config是实例化后的类
    logger = config.get_logger('train')

    # setup data_loader instances       #返回实例化后的，举例：HDF5DataLoader(*args, **module_args)
    data_loader = config.init_obj('data_loader', module_data)#取data_loader的type
    valid_data_loader = config.init_obj('valid_data_loader', module_data)
    # init_obj（data_loader的args也传入)→HDF5DataLoader→SequenceDataset→DynamicH5Dataset(config中有误？)→BaseVoxelDataset→Dataset

    # build model architecture, then print to console
    model = config.init_obj('arch', module_arch)
    logger.info(model)

    # init loss classes                     #用列表存储损失函数
    loss_ftns = [getattr(module_loss, loss)(**kwargs) for loss, kwargs in config['loss_ftns'].items()]

    # build optimizer, learning rate scheduler. delete every lines containing lr_scheduler for disabling scheduler
    trainable_params = filter(lambda p: p.requires_grad, model.parameters())    #需要优化的参数
    optimizer = config.init_obj('optimizer', torch.optim, trainable_params)     #损失函数

    lr_scheduler = config.init_obj('lr_scheduler', torch.optim.lr_scheduler, optimizer)

    trainer = Trainer(model, loss_ftns, optimizer,
                      config=config,
                      data_loader=data_loader,
                      valid_data_loader=valid_data_loader,
                      lr_scheduler=lr_scheduler)

    trainer.train()


if __name__ == '__main__':
    args = argparse.ArgumentParser(description='PyTorch Template')
    args.add_argument('-c', '--config', default=None, type=str,
                      help='config file path (default: None)')
    args.add_argument('-r', '--resume', default=None, type=str,
                      help='path to latest checkpoint (default: None)')
    args.add_argument('-d', '--device', default=None, type=str,
                      help='indices of GPUs to enable (default: all)')
    args.add_argument('--limited_memory', default=False, action='store_true',
                      help='prevent "too many open files" error by setting pytorch multiprocessing to "file_system".')

    # custom cli options to modify configuration from default values given in json file.
    CustomArgs = collections.namedtuple('CustomArgs', 'flags type target')#具名元组，我们可以像访问对象属性一样访问元组的元素
    options = [         #分别对应   flags           type        target
        CustomArgs(['--lr', '--learning_rate'], type=float, target='optimizer;args;lr'),
        CustomArgs(['--bs', '--batch_size'], type=int, target='data_loader;args;batch_size'),
        CustomArgs(['--rmb', '--reset_monitor_best'], type=bool, target='trainer;reset_monitor_best'),
        CustomArgs(['--vo', '--valid_only'], type=bool, target='trainer;valid_only')
    ]
    config = ConfigParser.from_args(args, options)  #将options添加到args，创建一个 ConfigParser 类的对象

    if args.parse_args().limited_memory:
        # https://github.com/pytorch/pytorch/issues/11201#issuecomment-421146936
        import torch.multiprocessing
        torch.multiprocessing.set_sharing_strategy('file_system')

    main(config)
