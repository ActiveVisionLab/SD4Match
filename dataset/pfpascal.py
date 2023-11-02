r""" PF-PASCAL dataset """

import os

import scipy.io as sio
import pandas as pd
import numpy as np
import torch
import matplotlib.pyplot as plt

from torch.utils.data import Dataset
from .dataset import CorrespondenceDataset
from utils.geometry import *

class PFPascalDataset(CorrespondenceDataset):

    def __init__(self, cfg, split, category="all"):
        """ PF-PASCAL dataset constructor """
        super(PFPascalDataset, self).__init__(cfg, split=split)

        self.cls = ['aeroplane', 'bicycle', 'bird', 'boat', 'bottle',
                    'bus', 'car', 'cat', 'chair', 'cow',
                    'diningtable', 'dog', 'horse', 'motorbike', 'person',
                    'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor']
        self.train_data = pd.read_csv(self.spt_path)
        
        if category != "all":
            if category in self.cls:
                self.src_imnames = []
                self.trg_imnames = []
                self.cls_ids = []
                self.flip = []
                for pair in self.train_data.iloc:
                    if self.cls[pair['class']-1] == category:
                        self.src_imnames.append(pair["source_image"])
                        self.trg_imnames.append(pair["target_image"])
                        self.cls_ids.append(pair['class']-1)
                        if split == "trn":
                            self.flip.append(pair["flip"])
                        else:
                            self.flip.append(0)
                self.src_imnames = np.array(self.src_imnames)
                self.trg_imnames = np.array(self.trg_imnames)
                self.cls_ids = np.array(self.cls_ids)
                self.flip = np.array(self.flip)
            else:
                raise ValueError(f"{category} is not in category option.")  
        else:
            self.src_imnames = np.array(self.train_data.iloc[:, 0])
            self.trg_imnames = np.array(self.train_data.iloc[:, 1])
            
            self.cls_ids = self.train_data.iloc[:, 2].values.astype('int') - 1
            self.split = split
            
            if split == 'trn':
                self.flip = self.train_data.iloc[:, 3].values.astype('int')
            else:
                self.flip = np.zeros((len(self.src_imnames),), dtype=np.int64)

        self.src_kps = []
        self.trg_kps = []
        self.src_bbox = []
        self.trg_bbox = []
        for src_imname, trg_imname, cls in zip(self.src_imnames, self.trg_imnames, self.cls_ids):
            src_anns = os.path.join(self.ann_path, self.cls[cls],
                                    os.path.basename(src_imname))[:-4] + '.mat'
            trg_anns = os.path.join(self.ann_path, self.cls[cls],
                                    os.path.basename(trg_imname))[:-4] + '.mat'

            src_kp = torch.tensor(read_mat(src_anns, 'kps')).float()
            trg_kp = torch.tensor(read_mat(trg_anns, 'kps')).float()
            src_box = torch.tensor(read_mat(src_anns, 'bbox')[0].astype(float))
            trg_box = torch.tensor(read_mat(trg_anns, 'bbox')[0].astype(float))

            src_kps = []
            trg_kps = []
            for src_kk, trg_kk in zip(src_kp, trg_kp):
                if len(torch.nonzero(torch.isnan(src_kk))) != 0 or \
                        len(torch.nonzero(torch.isnan(trg_kk))) != 0:
                    continue
                else:
                    src_kps.append(src_kk)
                    trg_kps.append(trg_kk)
            self.src_kps.append(torch.stack(src_kps).t())
            self.trg_kps.append(torch.stack(trg_kps).t())
            self.src_bbox.append(src_box)
            self.trg_bbox.append(trg_box)

        self.src_imnames = list(map(lambda x: os.path.basename(x), self.src_imnames))
        self.trg_imnames = list(map(lambda x: os.path.basename(x), self.trg_imnames))

        self.src_identifiers = [f"{self.cls[ids]}-{name[:-4]}-{flip}" for ids, name, flip in zip(self.cls_ids, self.src_imnames, self.flip)]
        self.trg_identifiers = [f"{self.cls[ids]}-{name[:-4]}-{flip}" for ids, name, flip in zip(self.cls_ids, self.trg_imnames, self.flip)]


    def __getitem__(self, idx):
        r""" Constructs and returns a batch for PF-PASCAL dataset """
        batch = super(PFPascalDataset, self).__getitem__(idx)
        
        h1, w1 = batch['src_img'].shape[1:]
        h2, w2 = batch['trg_img'].shape[1:]

        # Object bounding-box (resized following self.img_size)
        batch['src_bbox'] = self.get_bbox(self.src_bbox, idx, batch['src_imsize'], (h1, w1))
        batch['trg_bbox'] = self.get_bbox(self.trg_bbox, idx, batch['trg_imsize'], (h2, w2))
        # batch['pckthres'] = self.get_pckthres(batch)

        # Horizontal flipping key-points during training
        if self.flip[idx]:
            self.horizontal_flip(batch)
        batch['flip'] = torch.tensor([self.flip[idx]])

        # get pckthres
        batch['trg_pckthres'] = self.get_pckthres({'trg_img': batch['trg_img'], 'trg_bbox': batch['trg_bbox']})     # pckthres if matching to trg_img
        batch['src_pckthres'] = self.get_pckthres({'trg_img': batch['src_img'], 'trg_bbox': batch['src_bbox']})     # pckthres if matching to src_img
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

    def get_bbox(self, bbox_list, idx, ori_imsize, scaled_imsize):
        r""" Returns object bounding-box """
        '''
        ori_imsize: in (w, h)
        scaled_imsize: in (h, w)
        '''
        bbox = bbox_list[idx].clone()
        bbox[0::2] *= (scaled_imsize[1] / ori_imsize[0])
        bbox[1::2] *= (scaled_imsize[0] / ori_imsize[1])
        return bbox

    def horizontal_flip(self, batch):
        tmp = batch['src_bbox'][0].clone()
        batch['src_bbox'][0] = batch['src_img'].size(2) - batch['src_bbox'][2]
        batch['src_bbox'][2] = batch['src_img'].size(2) - tmp

        tmp = batch['trg_bbox'][0].clone()
        batch['trg_bbox'][0] = batch['trg_img'].size(2) - batch['trg_bbox'][2]
        batch['trg_bbox'][2] = batch['trg_img'].size(2) - tmp

        batch['src_kps'][0][:batch['n_pts']] = batch['src_img'].size(2) - batch['src_kps'][0][:batch['n_pts']]
        batch['trg_kps'][0][:batch['n_pts']] = batch['trg_img'].size(2) - batch['trg_kps'][0][:batch['n_pts']]

        batch['src_img'] = torch.flip(batch['src_img'], dims=(2,))
        batch['trg_img'] = torch.flip(batch['trg_img'], dims=(2,))


def read_mat(path, obj_name):
    r""" Reads specified objects from Matlab data file. (.mat) """
    mat_contents = sio.loadmat(path)
    mat_obj = mat_contents[obj_name]

    return mat_obj


class PFPascalImageDataset(Dataset):

    def __init__(self, cfg, split, category, transform=None, tokenizer=None):
        super().__init__()

        category_options = ['aeroplane', 'bicycle', 'bird', 'boat', 'bottle',
                    'bus', 'car', 'cat', 'chair', 'cow',
                    'diningtable', 'dog', 'horse', 'motorbike', 'person',
                    'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor']
        
        self.cfg = cfg
        self.root = os.path.join(cfg.DATASET.ROOT, 'pf-pascal')
        self.transform = transform
        self.tokenizer = tokenizer

        if category not in category_options and category != "all":
            raise ValueError(f"{category} is not in category option.")

        if split in ["trn", "val", "test"]:
            pair_list = os.path.join(self.root, split+'_pairs.csv')
            image_pairs = pd.read_csv(pair_list)
        else:
            raise ValueError(f"Invalid split {split}")

        unique = []
        for pair in image_pairs.iloc:
            src = pair['source_image'].split("/")[-1][:-4]
            trg = pair['target_image'].split("/")[-1][:-4]
            if split == "trn":
                flip = pair['flip']
            else:
                flip = 0
            cls = category_options[pair["class"]-1]
            if cls == category or category == "all":
                unique.append(cls + '-' + src + '-' + str(flip))
                unique.append(cls + '-' + trg + '-' + str(flip))
        unique = list(set(unique))
        unique.sort()
        
        # extract the 
        self.imnames = unique.copy()
        self.impaths = list(map(lambda x: os.path.join(self.root, 'PF-dataset-PASCAL/JPEGImages', x.split('-')[1]+'.jpg'), unique))
        self.category = list(map(lambda x: x.split('-')[0], unique))


    def __len__(self):

        return len(self.imnames)
    

    def __getitem__(self, index):
        
        output = {}

        img = plt.imread(self.impaths[index])
        img = np.copy(img)

        if self.transform is not None:
            img = self.transform(img)
            if int(self.imnames[index][-1]):
                img = torch.flip(img, (2,))

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