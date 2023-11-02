r""" PF-WILLOW dataset """

import os

import pandas as pd
import numpy as np
import torch
import matplotlib.pyplot as plt

from torch.utils.data import Dataset
from .dataset import CorrespondenceDataset
from utils.geometry import *

class PFWillowDataset(CorrespondenceDataset):

    def __init__(self, cfg, split, category):
        """PF-WILLOW dataset constructor"""
        super(PFWillowDataset, self).__init__(cfg, split=split)

        self.cls = ['car(G)', 'car(M)', 'car(S)', 'duck(S)',
                    'motorbike(G)', 'motorbike(M)', 'motorbike(S)',
                    'winebottle(M)', 'winebottle(wC)', 'winebottle(woC)']
        self.train_data = pd.read_csv(self.spt_path)

        if category == "all":
            self.src_imnames = np.array(self.train_data.iloc[:, 0])
            self.trg_imnames = np.array(self.train_data.iloc[:, 1])
            self.src_kps = self.train_data.iloc[:, 2:22].values
            self.trg_kps = self.train_data.iloc[:, 22:].values
        else:
            self.src_imnames, self.trg_imnames = [], []
            self.src_kps, self.trg_kps = [], []
            for pair in self.train_data.iloc:
                if category in pair["imageA"]:
                    self.src_imnames.append(pair["imageA"])
                    self.trg_imnames.append(pair["imageB"])
                    self.src_kps.append(pair[2:22].values)
                    self.trg_kps.append(pair[22:].values)
            self.src_imnames = np.stack(self.src_imnames, axis=0)
            self.trg_imnames = np.stack(self.trg_imnames, axis=0)
            self.src_kps = np.stack(self.src_kps, axis=0)
            self.trg_kps = np.stack(self.trg_kps, axis=0)

        self.cls_ids = list(map(lambda names: self.cls.index(names.split('/')[1]), self.src_imnames))
        self.src_imnames = list(map(lambda x: os.path.join(*x.split('/')[1:]), self.src_imnames))
        self.trg_imnames = list(map(lambda x: os.path.join(*x.split('/')[1:]), self.trg_imnames))
        
        self.src_identifiers = [f"{self.cls[ids]}-{name.split('/')[1][:-4]}" for ids, name in zip(self.cls_ids, self.src_imnames)]
        self.trg_identifiers = [f"{self.cls[ids]}-{name.split('/')[1][:-4]}" for ids, name in zip(self.cls_ids, self.trg_imnames)]

    def __getitem__(self, idx):
        """ Constructs and returns a batch for PF-WILLOW dataset """
        batch = super(PFWillowDataset, self).__getitem__(idx)
        
        h1, w1 = batch['src_img'].shape[1:]
        h2, w2 = batch['trg_img'].shape[1:]

        # get pckthres
        batch['trg_pckthres'] = self.get_pckthres({'trg_img': batch['trg_img'], 'trg_kps': batch['trg_kps'], 'n_pts': batch['n_pts']})     # pckthres if matching to trg_img
        batch['src_pckthres'] = self.get_pckthres({'trg_img': batch['src_img'], 'trg_kps': batch['src_kps'], 'n_pts': batch['n_pts']})     # pckthres if matching to src_img
        # default pckthres is the trg_pckthres, as by default we are matching in src->trg order
        batch['pckthres'] = batch['trg_pckthres'].clone()

        # convert the src_kps and trg_kps from 2 x N to N x 2
        batch['src_kps'] = batch['src_kps'].permute(1, 0)
        batch['trg_kps'] = batch['trg_kps'].permute(1, 0)

        # regularize coordinate 
        n_pts = batch['n_pts'] 
        batch['src_kps'][:n_pts] = regularise_coordinates(batch['src_kps'][:n_pts], h1, w1, eps=1e-4) 
        batch['trg_kps'][:n_pts] = regularise_coordinates(batch['trg_kps'][:n_pts], h2, w2, eps=1e-4) 

        # create src_identifier and trg_identifier to uniquely identify image
        batch['src_identifier'] = self.src_identifiers[idx]
        batch['trg_identifier'] = self.trg_identifiers[idx]

        return batch

    def get_pckthres(self, batch):
        """ Computes PCK threshold """
        if self.thres == 'bbox':
            npt = batch['n_pts']
            return torch.max(batch['trg_kps'][:, :npt].max(1)[0] - batch['trg_kps'][:, :npt].min(1)[0]).clone()
        elif self.thres == 'img':
            return torch.tensor(max(batch['trg_img'].size()[1], batch['trg_img'].size()[2]))
        else:
            raise Exception('Invalid pck evaluation level: %s' % self.thres)

    def get_points(self, pts_list, idx, ori_imsize, scaled_imsize):
        """ Returns key-points of an image """
        '''
        ori_imsize: in (w, h)
        scaled_imsize: in (h, w)
        '''
        point_coords = pts_list[idx, :].reshape(2, 10)
        point_coords = torch.tensor(point_coords.astype(np.float32))
        xy, n_pts = point_coords.size()
        pad_pts = torch.zeros((xy, self.max_pts - n_pts)) - 2
        x_crds = point_coords[0] * (scaled_imsize[1] / ori_imsize[0])
        y_crds = point_coords[1] * (scaled_imsize[0] / ori_imsize[1])
        kps = torch.cat([torch.stack([x_crds, y_crds]), pad_pts], dim=1)

        return kps, n_pts
    

class PFWillowImageDataset(Dataset):

    def __init__(self, cfg, split, category, transform=None, tokenizer=None):
        super().__init__()

        category_options = ['car(G)', 'car(M)', 'car(S)', 'duck(S)',
                    'motorbike(G)', 'motorbike(M)', 'motorbike(S)',
                    'winebottle(M)', 'winebottle(wC)', 'winebottle(woC)']
        
        self.cfg = cfg
        self.root = os.path.join(cfg.DATASET.ROOT, 'pf-willow')
        self.transform = transform
        self.tokenizer = tokenizer

        if category not in category_options and category != "all":
            raise ValueError(f"{category} is not in category option.")

        pair_list = os.path.join(self.root, 'test_pairs.csv')
        image_pairs = pd.read_csv(pair_list)


        unique = []
        for pair in image_pairs.iloc:
            src = pair['imageA'].split("/")[-1][:-4]
            trg = pair['imageB'].split("/")[-1][:-4]
            cls = pair['imageA'].split("/")[1]
            if cls == category or category == "all":
                unique.append(cls + '-' + src)
                unique.append(cls + '-' + trg)
        unique = list(set(unique))
        unique.sort()
        
        # extract the 
        self.imnames = unique.copy()
        self.impaths = list(map(lambda x: os.path.join(self.root, 'PF-dataset', x.split('-')[0], x.split('-')[1]+'.png'), unique))
        self.category = list(map(lambda x: x.split('-')[0], unique))


    def __len__(self):

        return len(self.imnames)
    

    def __getitem__(self, index):
        
        output = {}

        img = plt.imread(self.impaths[index])
        img = np.copy(img)

        if self.transform is not None:
            img = self.transform(img)

        output.update({"pixel_values": img})

        category = self.category[index]
        caption = f"a photo of a {category}"

        if self.tokenizer is not None:
            inputs = self.tokenizer(
            caption, max_length=self.tokenizer.model_max_length, padding="max_length", truncation=True, return_tensors="pt"
        )
            inputs = inputs.input_ids

            output.update({"input_ids": inputs[0]})
        else:
            output.update({"prompt": caption})

        output.update({"identifier": self.imnames[index], "impath": self.impaths[index], "category": category})

        return output
