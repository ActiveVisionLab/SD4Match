r""" Superclass for semantic correspondence datasets """

import os

from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
import torch


class CorrespondenceDataset(Dataset):
    r""" Parent class of PFPascal, PFWillow, and SPair """
    def __init__(self, cfg, split, thres="auto"):
        '''
        benchmark: pfwillow, pfpascal, spair.
        img_size: image will be resized to this size。
        datapath: path to the benchmark folder.
        thres: bbox or img, the length used to measure pck.
        split: trn, test or val.
        '''
        """ CorrespondenceDataset constructor """
        super(CorrespondenceDataset, self).__init__()

        # {Directory name, Layout path, Image path, Annotation path, PCK threshold}
        self.metadata = {
            'pfwillow': ('pf-willow',
                         'test_pairs.csv',
                         'PF-dataset',
                         '',
                         'bbox'),
            'pfpascal': ('pf-pascal',
                         '_pairs.csv',
                         'PF-dataset-PASCAL/JPEGImages',
                         'PF-dataset-PASCAL/Annotations',
                         'img'),
            'spair':    ('SPair-71k',
                         'Layout/large',
                         'JPEGImages',
                         'PairAnnotation',
                         'bbox')
        }

        benchmark = cfg.DATASET.NAME
        datapath = cfg.DATASET.ROOT
        img_size = cfg.DATASET.IMG_SIZE
        norm_mean = cfg.DATASET.MEAN
        norm_std = cfg.DATASET.STD

        self.original_imgsize = False
        if split == 'test_ori':
            split = 'test'
            self.original_imgsize = True

        # Directory path for train, val, or test splits
        base_path = os.path.join(os.path.abspath(datapath), self.metadata[benchmark][0])
        if benchmark == 'pfpascal':
            self.spt_path = os.path.join(base_path, split+'_pairs.csv')
        elif benchmark == 'spair':
            self.spt_path = os.path.join(base_path, self.metadata[benchmark][1], split+'.txt')
        elif benchmark == 'pfwillow':
            self.spt_path = os.path.join(base_path, self.metadata[benchmark][1])
        else:
            raise ValueError('benchmark must be within pfpascal, spair and pfwillow')

        # Directory path for images
        self.img_path = os.path.join(base_path, self.metadata[benchmark][2])

        # Directory path for annotations
        if benchmark == 'spair':
            self.ann_path = os.path.join(base_path, self.metadata[benchmark][3], split)
        else:
            self.ann_path = os.path.join(base_path, self.metadata[benchmark][3])

        # Miscellaneous
        self.max_pts = 20
        self.split = split
        self.img_size = (img_size, img_size)
        self.benchmark = benchmark
        self.range_ts = torch.arange(self.max_pts)
        self.thres = self.metadata[benchmark][4] if thres == 'auto' else thres
        if self.original_imgsize:
            self.transform = transforms.Compose([
                             transforms.ToTensor(),
                             transforms.Normalize(mean=norm_mean, std=norm_std)
                            ])
        else:
            self.transform = transforms.Compose([
                             transforms.Resize((self.img_size[0], self.img_size[1])),
                             transforms.ToTensor(),
                             transforms.Normalize(mean=norm_mean, std=norm_std)
                            ])
        

        # To get initialized in subclass constructors
        self.train_data = []
        self.src_imnames = []
        self.trg_imnames = []
        self.cls = []
        self.cls_ids = []
        self.src_kps = []
        self.trg_kps = []

    def __len__(self):
        r""" Returns the number of pairs """
        return len(self.src_imnames)

    def __getitem__(self, idx):
        r""" Constructs and return a batch """

        # Image name
        batch = dict()
        batch['src_imname'] = self.src_imnames[idx]
        batch['trg_imname'] = self.trg_imnames[idx]

        # Object category
        batch['category_id'] = self.cls_ids[idx]
        batch['category'] = self.cls[batch['category_id']]

        # Image as numpy (original width, original height)
        src_pil = self.get_image(self.src_imnames, idx)
        trg_pil = self.get_image(self.trg_imnames, idx)
        batch['src_imsize'] = src_pil.size
        batch['trg_imsize'] = trg_pil.size

        # Image as tensor
        batch['src_img'] = self.transform(src_pil)
        batch['trg_img'] = self.transform(trg_pil)

        h1, w1 = batch['src_img'].shape[1:]
        h2, w2 = batch['trg_img'].shape[1:]

        # Key-points (re-scaled)
        batch['src_kps'], num_pts = self.get_points(self.src_kps, idx, src_pil.size, (h1, w1))
        batch['trg_kps'], _ = self.get_points(self.trg_kps, idx, trg_pil.size, (h2, w2))
        batch['n_pts'] = torch.tensor(num_pts)

        # Total number of pairs in training split
        batch['datalen'] = len(self.train_data)

        return batch

    def get_image(self, imnames, idx):
        r""" Reads PIL image from path """
        path = os.path.join(self.img_path, imnames[idx])
        return Image.open(path).convert('RGB')

    def get_pckthres(self, batch):
        r""" Computes PCK threshold """
        if self.thres == 'bbox':
            bbox = batch['trg_bbox'].clone()
            if len(bbox.shape) == 2:
                bbox = bbox.squeeze(0)
            bbox_w = (bbox[2] - bbox[0])
            bbox_h = (bbox[3] - bbox[1])
            pckthres = torch.max(bbox_w, bbox_h)
        elif self.thres == 'img':
            imsize_t = batch['trg_img'].size()
            if len(imsize_t) == 4:
                imsize_t = imsize_t[1:]
            pckthres = torch.tensor(max(imsize_t[1], imsize_t[2]))
        else:
            raise Exception('Invalid pck threshold type: %s' % self.thres)
        return pckthres.float()

    def get_points(self, pts_list, idx, ori_imsize, scaled_imsize):
        r""" Returns key-points of an image """
        '''
        ori_imsize: in (w, h)
        scaled_imsize: in (h, w)
        '''
        xy, n_pts = pts_list[idx].size()
        pad_pts = torch.zeros((xy, self.max_pts - n_pts)) - 2
        x_crds = pts_list[idx][0] * (scaled_imsize[1] / ori_imsize[0])
        y_crds = pts_list[idx][1] * (scaled_imsize[0] / ori_imsize[1])
        kps = torch.cat([torch.stack([x_crds, y_crds]), pad_pts], dim=1)

        return kps, n_pts
