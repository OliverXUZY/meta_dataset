import os
import random
import h5py
from PIL import Image
import json
#import cv2
import numpy as np

from .datasets import register
from .transforms import *
from torchvision.datasets import ImageFolder


import torch
from torch.utils.data import Dataset
from .meta_dataset import config as config_lib
from .meta_dataset import sampling
from .meta_dataset.utils import Split
from .meta_dataset.transform import get_transforms
from .meta_dataset import dataset_spec as dataset_spec_lib

@register('meta-dataset') 
class MetaDatasetH5(torch.utils.data.Dataset):
    def __init__(self, root = None, split='train', size=224, transform=None,
               n_batch=200, n_episode=4, n_way=5, n_shot=1, n_query=15, deterministic = False, 
               domains = ['quickdraw', 'omniglot', 'ilsvrc_2012', 'aircraft', 'fungi', 'cu_birds', 
                              'vgg_flower']):
        """
        Args:
        root (str): root path of dataset.
        split (str): dataset split. Default: 'train'
        size (int): image resolution. Default: 84
        transform (str): data augmentation. Default: None
        """

        split_dict = {'train': 'train',        # standard train
                  'val': 'val',            # standard val
                  'test': 'test',          # standard test
                  'meta-train': 'train',   # meta-train
                  'meta-val': 'val',                   # meta-val
                  'meta-test': 'test',                 # meta-test
                 }
        
        split = split_dict.get(split) or split
        if split == "train":
            split = Split.TRAIN
            datasets = domains
        else:
            split = Split.TEST
            datasets = domains
        
        root_path = "/datadrive2/datasets/meta_dataset_h5"
        # dataset specifications
        all_dataset_specs = []
        for dataset_name in datasets:
            dataset_records_path = os.path.join(root_path, dataset_name)
            dataset_spec = dataset_spec_lib.load_dataset_spec(dataset_records_path)
            all_dataset_specs.append(dataset_spec)
        
        self.n_class = sum([len(d_spec.get_classes(split=split)) for d_spec in all_dataset_specs])

        self.statistics = {'mean': [0.485, 0.456, 0.406],
                       'std': [0.229, 0.224, 0.225]}
    
        self.transforms = get_transform(transform, size, self.statistics)

        self.datasets = datasets
        self.len = n_episode * len(datasets) # NOTE: not all datasets get equal number of episodes per epoch

        self.class_map = {} # 2-level dict of h5 paths
        self.class_h5_dict = {} # 2-level dict of opened h5 files
        self.class_samplers = {} # 1-level dict of samplers, one for each dataset
        self.class_images = {} # 2-level dict of image ids, one list for each class

        for i, dataset_name in enumerate(datasets):
            dataset_spec = all_dataset_specs[i]
            base_path = dataset_spec.path
            class_set = dataset_spec.get_classes(split) # class ids in this split
            num_classes = len(class_set)

            record_file_pattern = dataset_spec.file_pattern
            assert record_file_pattern.startswith('{}'), f'Unsupported {record_file_pattern}.'

            self.class_map[dataset_name] = {}
            self.class_h5_dict[dataset_name] = {}
            self.class_images[dataset_name] = {}

            for class_id in class_set:
                data_path = os.path.join(base_path, record_file_pattern.format(class_id))
                self.class_map[dataset_name][class_id] = data_path.replace('tfrecords', 'h5')
                self.class_h5_dict[dataset_name][class_id] = None # closed h5 is None
                self.class_images[dataset_name][class_id] = [str(j) for j in range(dataset_spec.get_total_images_per_class(class_id))]

        self.n_batch = n_batch
        self.n_episode = n_episode
        self.n_way = n_way
        self.n_shot = n_shot
        self.n_query = n_query
        self.deterministic = deterministic
    
    def __len__(self):
        return self.n_batch * self.n_episode
    
    def get_next(self, source, class_id, idx):
        # fetch h5 path
        h5_path = self.class_map[source][class_id]

        # load h5 file if None
        if self.class_h5_dict[source][class_id] is None: # will be closed in the end of main.py
            self.class_h5_dict[source][class_id] = h5py.File(h5_path, 'r')

        h5_file = self.class_h5_dict[source][class_id]
        record = h5_file[idx]
        x = record['image'][()]

        if self.transforms:
            x = Image.fromarray(x)
            x = self.transforms(x)

        return x
    
    def __getitem__(self, idx, source = 'aircraft'):
        if self.deterministic:
            np.random.seed(idx)

        # Randomly select a source (domain)
        source = np.random.choice(self.datasets, 1)[0]

        # Number of support and query images
        s, q = self.n_shot, self.n_query

        # Sampled classes and images
        sampled_classes = np.random.choice(list(self.class_images[source].keys()), self.n_way, replace=False)
        support_images, query_images, support_labels, query_labels = [], [], [], []

        for class_id in sampled_classes:
            # Get all available image indices for the class
            available_indices = self.class_images[source][class_id]

            while len(available_indices) < s + q:  # If not enough images, resample class
                class_id = np.random.choice(list(self.class_images[source].keys()))
                available_indices = self.class_images[source][class_id]

            # Randomly sample images for support and query
            sampled_indices = np.random.choice(available_indices, s + q, replace=False)
            
            # Split into support and query samples
            s_indices, q_indices = sampled_indices[:s], sampled_indices[s:]

            # Fetch images and append to respective lists
            for i in s_indices:
                img = self.get_next(source, class_id, i)
                support_images.append(img)
                support_labels.append(class_id)

            for i in q_indices:
                img = self.get_next(source, class_id, i)
                query_images.append(img)
                query_labels.append(class_id)

        # Convert lists to torch tensors
        support_images = torch.stack(support_images, dim=0)
        query_images = torch.stack(query_images, dim=0)
        support_labels = torch.tensor(support_labels)
        query_labels = torch.tensor(query_labels)

        support_images = support_images.unsqueeze(0).unsqueeze(2)
        query_images = query_images.unsqueeze(0)

        return support_images, query_images, query_labels

