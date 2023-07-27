import os
import pandas as pd
from tqdm import tqdm
from torch.utils.data import ConcatDataset


data_sources = ('esim', 'ijrr', 'mvsec', 'eccd', 'hqfd', 'unknown')
# Usage: name = data_sources[1], idx = data_sources.index('ijrr')

def concatenate_datasets(data_file, dataset_type, dataset_kwargs={}):
    """
    Generates a dataset for each data_path specified in data_file and concatenates the datasets.
    :param data_file: A file containing a list of paths to CTI h5 files.
                      Each file is expected to have a sequence of frame_{:09d}
    :param dataset_type: Pointer to dataset class
    :param sequence_length: Desired length of each sequence
    :return ConcatDataset: concatenated dataset of all data_paths in data_file
    concatenate_datasets(data_file, SequenceDataset, sequence_kwargs)
    """                                                         #.tolist() 方法：用于将数组或列表转换为普通的 Python 列表
    data_paths = pd.read_csv(data_file,header=None).values.flatten().tolist()
    dataset_list = []
    time=0
    for data_path in tqdm(data_paths):  #tqdm显示进度条
        dataset_list.append(dataset_type(data_path, **dataset_kwargs))#dataset_type是SequenceDataset,返回实例化后的SequenceDataset
    return ConcatDataset(dataset_list)

