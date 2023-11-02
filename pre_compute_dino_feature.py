import os
from tqdm import tqdm

import torch
import torchvision.transforms as T
from transformers import AutoModel, AutoImageProcessor

from dataset import SPairImageDataset, PFPascalImageDataset, PFWillowImageDataset

from config import get_default_defaults
from PIL import Image

cfg = get_default_defaults()
cfg.merge_from_file("config/dift.py")

transforms = T.Compose([
    T.ToTensor(),
    T.Resize((cfg.DATASET.IMG_SIZE, cfg.DATASET.IMG_SIZE)),
    T.Normalize(mean=cfg.DATASET.MEAN, std=cfg.DATASET.STD)
])


for dataset in ["pfpascal", "pfwillow", "spair"]:

    output_dict = {}

    for split in ["trn", "val", "test"]:

        print(f"Compute DINOv2 Feature of {dataset} {split}")

        # PF-Willow dataset has only testing data, handle this exception:
        if dataset == "pfwillow" and split != "test":
            continue

        cfg.DATASET.NAME = dataset

        if cfg.DATASET.NAME == 'spair':
            img_dataset = SPairImageDataset(cfg, split, "all", transforms)
        elif cfg.DATASET.NAME == 'pfpascal':
            img_dataset = PFPascalImageDataset(cfg, split, "all", transforms)
        elif cfg.DATASET.NAME == 'pfwillow':
            img_dataset = PFWillowImageDataset(cfg, split, "all", transforms)

        sd = "2-1"
        output_dir = cfg.FEATURE_EXTRACTOR.ASSET_ROOT

        output_dir = os.path.join(output_dir, "DINOv2", cfg.DATASET.NAME)
        vision_encoder = AutoModel.from_pretrained('facebook/dinov2-base')
        vision_processor = AutoImageProcessor.from_pretrained('facebook/dinov2-base')
        
        for param in vision_encoder.parameters():
            param.requires_grad = False
        vision_encoder.to("cuda")

        for i in tqdm(range(len(img_dataset))):
            impath = img_dataset.impaths[i]
            imname = img_dataset.imnames[i]

            img = Image.open(impath)            
            if cfg.DATASET.NAME == "pfpascal" and int(imname[-1]):
                img = img.transpose(Image.FLIP_LEFT_RIGHT)

            img = vision_processor(img, return_tensors="pt")["pixel_values"]
            encoder_output = vision_encoder(pixel_values=img.to(vision_encoder.device))
            output_dict.update({imname: encoder_output})

    os.makedirs(output_dir, exist_ok=True)
    torch.save(output_dict, os.path.join(output_dir, "cached_output.pt"))