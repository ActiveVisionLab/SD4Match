r""" SPair-71k dataset """

import json
import glob
import os
from tqdm import tqdm

import torch.nn.functional as F
import torch
from torch.utils.data import Dataset
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

from .dataset import CorrespondenceDataset
from utils.geometry import *


class SPairDataset(CorrespondenceDataset):

    def __init__(self, cfg, split, category='all'):
        r""" SPair-71k dataset constructor """
        super(SPairDataset, self).__init__(cfg, split=split)

        category_options = ["aeroplane", "bicycle", "bird", "boat", "bottle", "bus",
                            "car", "cat", "chair", "cow", "dog", "horse", "motorbike",
                            "person", "pottedplant", "sheep", "train", "tvmonitor"]

        self.train_data = open(self.spt_path).read().split('\n')
        self.train_data = self.train_data[:len(self.train_data) - 1]

        if category != "all":
            if category in category_options:
                train_data = []
                for pair in self.train_data:
                    if category in pair:
                        train_data.append(pair)
                self.train_data = train_data
            else:
                raise ValueError(f"{category} is not in category option.")                

        self.src_imnames = list(map(lambda x: x.split('-')[1] + '.jpg', self.train_data))
        self.trg_imnames = list(map(lambda x: x.split('-')[2].split(':')[0] + '.jpg', self.train_data))
        self.seg_path = os.path.abspath(os.path.join(self.img_path, os.pardir, 'Segmentation'))
        self.cls = os.listdir(self.img_path)
        self.cls.sort()

        anntn_files = []
        for data_name in self.train_data:
            anntn_files.append(glob.glob('%s/%s.json' % (self.ann_path, data_name))[0])
        # anntn_files = list(map(lambda x: json.load(open(x)), anntn_files))

        # self.src_kps = list(map(lambda x: torch.tensor(x['src_kps']).t().float(), anntn_files))
        # self.trg_kps = list(map(lambda x: torch.tensor(x['trg_kps']).t().float(), anntn_files))
        # self.src_bbox = list(map(lambda x: torch.tensor(x['src_bndbox']).float(), anntn_files))
        # self.trg_bbox = list(map(lambda x: torch.tensor(x['trg_bndbox']).float(), anntn_files))
        # self.cls_ids = list(map(lambda x: self.cls.index(x['category']), anntn_files))

        # self.vpvar = list(map(lambda x: torch.tensor(x['viewpoint_variation']), anntn_files))
        # self.scvar = list(map(lambda x: torch.tensor(x['scale_variation']), anntn_files))
        # self.trncn = list(map(lambda x: torch.tensor(x['truncation']), anntn_files))
        # self.occln = list(map(lambda x: torch.tensor(x['occlusion']), anntn_files))

        self.src_kps, self.trg_kps, self.src_bbox, self.trg_bbox, self.cls_ids = [], [], [], [], []
        self.vpvar, self.scvar, self.trncn, self.occln = [], [], [], []
        print("Reading SPair-71k information...")
        for anntn_file in tqdm(anntn_files):
            anntn = json.load(open(anntn_file))
            self.src_kps.append(torch.tensor(anntn['src_kps']).t().float())
            self.trg_kps.append(torch.tensor(anntn['trg_kps']).t().float())
            self.src_bbox.append(torch.tensor(anntn['src_bndbox']).float())
            self.trg_bbox.append(torch.tensor(anntn['trg_bndbox']).float())
            self.cls_ids.append(self.cls.index(anntn['category']))

            self.vpvar.append(torch.tensor(anntn['viewpoint_variation']))
            self.scvar.append(torch.tensor(anntn['scale_variation']))
            self.trncn.append(torch.tensor(anntn['truncation']))
            self.occln.append(torch.tensor(anntn['occlusion']))

        self.src_identifiers = [f"{self.cls[ids]}-{name[:-4]}" for ids, name in zip(self.cls_ids, self.src_imnames)]
        self.trg_identifiers = [f"{self.cls[ids]}-{name[:-4]}" for ids, name in zip(self.cls_ids, self.trg_imnames)]

    def __len__(self):
        return len(self.src_imnames)

    def __getitem__(self, idx):
        r""" Construct and return a batch for SPair-71k dataset """
        sample = super(SPairDataset, self).__getitem__(idx)

        h1, w1 = sample['src_img'].shape[1:]
        h2, w2 = sample['trg_img'].shape[1:]

        sample['src_mask'] = self.get_mask(sample, sample['src_imname'], (h1, w1))
        sample['trg_mask'] = self.get_mask(sample, sample['trg_imname'], (h2, w2))

        sample['src_bbox'] = self.get_bbox(self.src_bbox, idx, sample['src_imsize'], (h1, w1))
        sample['trg_bbox'] = self.get_bbox(self.trg_bbox, idx, sample['trg_imsize'], (h2, w2))

        # get pckthres
        sample['trg_pckthres'] = self.get_pckthres({'trg_img': sample['trg_img'], 'trg_bbox': sample['trg_bbox']})     # pckthres if matching to trg_img
        sample['src_pckthres'] = self.get_pckthres({'trg_img': sample['src_img'], 'trg_bbox': sample['src_bbox']})     # pckthres if matching to src_img
        # default pckthres is the trg_pckthres, as by default we are matching in src->trg order
        sample['pckthres'] = sample['trg_pckthres'].clone()

        # convert the src_kps and trg_kps from 2 x N to N x 2
        sample['src_kps'] = sample['src_kps'].permute(1, 0)
        sample['trg_kps'] = sample['trg_kps'].permute(1, 0)

        # regularize coordinate 
        n_pts = sample['n_pts'] 
        sample['src_kps'][:n_pts] = regularise_coordinates(sample['src_kps'][:n_pts], h1, w1, eps=1e-4) 
        sample['trg_kps'][:n_pts] = regularise_coordinates(sample['trg_kps'][:n_pts], h2, w2, eps=1e-4) 

        sample['vpvar'] = self.vpvar[idx]
        sample['scvar'] = self.scvar[idx]
        sample['trncn'] = self.trncn[idx]
        sample['occln'] = self.occln[idx]

        # create src_identifier and trg_identifier to uniquely identify image
        sample['src_identifier'] = self.src_identifiers[idx]
        sample['trg_identifier'] = self.trg_identifiers[idx]

        return sample
    

    def get_mask(self, sample, imname, scaled_imsize):
        '''
        scaled_imsize: in (h, w)
        '''
        mask_path = os.path.join(self.seg_path, sample['category'], imname.split('.')[0] + '.png')

        tensor_mask = torch.tensor(np.array(Image.open(mask_path)))

        class_dict = {'aeroplane': 0, 'bicycle': 1, 'bird': 2, 'boat': 3, 'bottle': 4,
                      'bus': 5, 'car': 6, 'cat': 7, 'chair': 8, 'cow': 9,
                      'diningtable': 10, 'dog': 11, 'horse': 12, 'motorbike': 13, 'person': 14,
                      'pottedplant': 15, 'sheep': 16, 'sofa': 17, 'train': 18, 'tvmonitor': 19}

        class_id = class_dict[sample['category']] + 1
        tensor_mask[tensor_mask != class_id] = 0
        tensor_mask[tensor_mask == class_id] = 255

        tensor_mask = F.interpolate(tensor_mask.unsqueeze(0).unsqueeze(0).float(),
                                    size=(scaled_imsize[0], scaled_imsize[1]),
                                    mode='bilinear', align_corners=True).int().squeeze()

        return tensor_mask

    def get_image(self, img_names, idx):
        r""" Return image tensor """
        path = os.path.join(self.img_path, self.cls[self.cls_ids[idx]], img_names[idx])

        return Image.open(path).convert('RGB')

    def get_pckthres(self, sample):
        r""" Compute PCK threshold """
        return super(SPairDataset, self).get_pckthres(sample)

    def get_points(self, pts_list, idx, ori_imsize, scaled_imsize):
        r""" Return key-points of an image """
        return super(SPairDataset, self).get_points(pts_list, idx, ori_imsize, scaled_imsize)

    def match_idx(self, kps, n_pts):
        r""" Sample the nearst feature (receptive field) indices """
        return super(SPairDataset, self).match_idx(kps, n_pts)

    def get_bbox(self, bbox_list, idx, ori_imsize, scaled_imsize):
        r""" Return object bounding-box """
        '''
        ori_imsize: in (w, h)
        scaled_imsize: in (h, w)
        '''
        bbox = bbox_list[idx].clone()
        bbox[0::2] *= (scaled_imsize[1] / ori_imsize[0])
        bbox[1::2] *= (scaled_imsize[0] / ori_imsize[1])
        return bbox
    

class SPairImageDataset(Dataset):

    def __init__(self, cfg, split, category, transform=None, tokenizer=None):
        super().__init__()

        category_options = ["aeroplane", "bicycle", "bird", "boat", "bottle", "bus",
                            "car", "cat", "chair", "cow", "dog", "horse", "motorbike",
                            "person", "pottedplant", "sheep", "train", "tvmonitor"]

        self.cfg = cfg
        self.root = os.path.join(cfg.DATASET.ROOT, 'SPair-71k')
        self.transform = transform
        self.tokenizer = tokenizer

        if category not in category_options and category != "all":
            raise ValueError(f"{category} is not in category option.")


        if split in ["trn", "val", "test"]:
            pair_list = os.path.join(self.root, 'Layout/large', split+'.txt')
            # extract unique image list from pair list
            with open(pair_list, 'r') as f:
                lines = f.readlines()
        elif split == "all":
            lines = []
            for s in ["trn", "val", "test"]:
                pair_list = os.path.join(self.root, 'Layout/large', s+'.txt')
                # extract unique image list from pair list
                with open(pair_list, 'r') as f:
                    lines += f.readlines()
    
        unique = []
        for line in lines:
            src = line.strip().split('-')[1]
            trg = line.strip().split('-')[2].split(':')[0]
            cls = line.strip().split('-')[2].split(':')[1]
            if cls == category or category == "all":
                unique.append(cls + '-' + src)
                unique.append(cls + '-' + trg)
        unique = list(set(unique))
        unique.sort()

        # extract the 
        get_name = lambda x: os.path.join(x.split('-')[0], x.split('-')[1])
        self.imnames = unique.copy()
        self.impaths = list(map(lambda x: os.path.join(self.root, 'JPEGImages', get_name(x)) + '.jpg', unique))
        self.annofiles = list(map(lambda x: os.path.join(self.root, 'ImageAnnotation', get_name(x)) + '.json', unique))

        anno_data = []
        for annofile in self.annofiles:
            anno_data.append(glob.glob(annofile)[0])
        anno_data = list(map(lambda x: json.load(open(x)), anno_data))

        self.bbox = list(map(lambda x: torch.tensor(x['bndbox']).float(), anno_data))   # in [x1, y1, x2, y2] format
        self.category = list(map(lambda x: x['category'], anno_data))

    
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


    def get_crop_size(self, bbox, img_size):
        '''
        IN:
            bbox [Tuple] (x1, y1, x2, y2)
            img_size [Tuple] (h, w)
        '''
        x1, y1, x2, y2 = bbox
        h, w = img_size

        crop_size = max((x2-x1), (y2-y1))
        if (x2-x1) == crop_size:
            y1 = y1 - (crop_size-(y2-y1))  / 2
            y2 = y2 + (crop_size-(y2-y1)) / 2
        elif (y2-y1) == crop_size:
            x1 = x1 - (crop_size-(x2-x1)) / 2
            x2 = x2 + (crop_size-(x2-x1)) / 2

        # check whether the crop size out of boundary
        x1 = max(0, x1)
        x2 = min(w, x2) 
        y1 = max(0, y1)
        y2 = min(h, y2)

        return int(x1), int(y1), int(x2), int(y2)