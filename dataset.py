# -*- coding: utf-8 -*-
"""
@author: fan weiquan
"""

import torch
import numpy as np
import os
from torch.utils.data import Dataset
from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
import torchvision

mytransform = torchvision.transforms.Compose([torchvision.transforms.Normalize([0.485, 0.456, 0.406],[0.229, 0.224, 0.225])])

class Custom_dataset(Dataset):
    def __init__(self, input_dir, names_labels, num_pairs):
        self.input_dir = input_dir
        self.names_labels = names_labels
        self.num_pairs = num_pairs

        self.pair_names = np.array(names_labels['name_pair'])
        self.names = np.array(names_labels['name'])
        self.labels = np.array(names_labels['label'])

    def __getitem__(self, index):
        path = os.path.join(self.input_dir, self.names[index] + '.png')
        image = Image.open(path).convert('RGB')
        image = torch.from_numpy(np.float32(image).transpose((2,0,1)).copy())
        image = np.float32(mytransform(image))

        name_pair = self.pair_names[index].split('-')
        image_pairs = []
        for i in range(min(self.num_pairs, len(name_pair))):
            name = name_pair[i]
            path_pair = os.path.join(self.input_dir, name + '.png')
            image_pair = Image.open(path_pair).convert('RGB')
            image_pair = torch.from_numpy(np.float32(image_pair).transpose((2,0,1)).copy())
            image_pair = np.float32(mytransform(image_pair))

            image_pairs.append(image_pair)
        if len(name_pair)==1:
            for i in range(self.num_pairs - 1):
                image_pairs.append(image_pair)

        label = torch.LongTensor([self.labels[index]]).squeeze()

        return image, image_pairs, label
    
    def __len__(self):
        return len(self.names_labels)


    # def collate_fn(self, data):
    #     dat = pd.DataFrame(data)
    #     return pad_sequence(dat[0], True), pad_sequence(dat[1], True), torch.LongTensor(dat[2].tolist())

