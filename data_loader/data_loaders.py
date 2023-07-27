from torch.utils.data import DataLoader
# local modules
from .dataset import DynamicH5Dataset, SequenceDataset
from utils.data import  concatenate_datasets

class InferenceDataLoader(DataLoader):

    def __init__(self, data_path, pin_memory=True, dataset_kwargs=None, ltype="H5"):
        if dataset_kwargs is None:
            dataset_kwargs = {}
        if ltype == "H5":
            dataset = DynamicH5Dataset(data_path, **dataset_kwargs)
        else:
            raise Exception("Unknown loader type {}".format(ltype))
        super().__init__(dataset, shuffle=False, pin_memory=pin_memory)


class HDF5DataLoader(DataLoader):
    """
    """
    def __init__(self, data_file, batch_size, shuffle=True, num_workers=1,
                 pin_memory=True, sequence_kwargs={}):
        dataset = concatenate_datasets(data_file, SequenceDataset, sequence_kwargs) #实例化后的SequenceDataset,用列表存储
        super().__init__(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, pin_memory=pin_memory)
        #执行父类的初始化函数，变成可迭代对象
