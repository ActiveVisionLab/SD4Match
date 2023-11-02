import os

import torch
import torchvision.transforms as T
from torch.utils.data import DataLoader

import argparse
from tqdm import tqdm
import numpy as np

from utils.misc import *
from utils.matching import *
from utils.geometry import *
from utils.evaluator import PCKEvaluator
from config import get_default_defaults
from dataset import PFPascalDataset, PFPascalImageDataset, SPairDataset, SPairImageDataset, PFWillowDataset, PFWillowImageDataset

from src.stable_diffusion.sd_feature_extractor import SDFeatureExtraction
from src.stable_diffusion.prompt import PromptManager
from src.stable_diffusion.hybrid_captioner import HybridCaptioner

os.chdir(os.path.dirname(os.path.realpath(__file__)))

parser = argparse.ArgumentParser()
parser.add_argument('--config_file', default='config/learnedToken.py', type=str)

parser.add_argument('--dataset', default='spair', type=str)
parser.add_argument('--split', default='test', type=str)
parser.add_argument('--category', default='all', type=str)
parser.add_argument('--prompt_type', default='text', type=str, help="choose between [text | empty | (learned prompt type)]")
parser.add_argument('--timestep', default=50, type=int)
parser.add_argument('--layer', default=1, type=int)

parser.add_argument('--batch_size', default=1, type=int)
parser.add_argument('--num_workers', default=2, type=int)
parser.add_argument('--device', default=0, type=int)

args = parser.parse_args()

np.random.seed(0)
torch.manual_seed(0)

torch.cuda.set_device(args.device)

cfg = get_default_defaults()
cfg.merge_from_file(args.config_file)
# override prompt type
cfg.DATASET.NAME = args.dataset
cfg.FEATURE_EXTRACTOR.PROMPT_TYPE = args.prompt_type
cfg.FEATURE_EXTRACTOR.SELECT_TIMESTEP = args.timestep
cfg.FEATURE_EXTRACTOR.SELECT_LAYER = args.layer

# create prompter to get prompt
if "CPM" in args.prompt_type:
    ckpt = torch.load(os.path.join(cfg.FEATURE_EXTRACTOR.PROMPT_CACHE_ROOT, args.prompt_type, "ckpt.pt"))
    captioner = HybridCaptioner(args.prompt_type, cfg.FEATURE_EXTRACTOR.ASSET_ROOT, "cuda")
    captioner.load_state_dict(ckpt, strict=False)
    captioner.to("cuda")
else:
    prompter = PromptManager(cfg)

# create stable diffusion pipeline
feature_extractor = SDFeatureExtraction(cfg)
feature_extractor = feature_extractor.to("cuda")

# create dataset and dataloader
transforms = T.Compose([
    T.ToTensor(),
    T.Resize((cfg.DATASET.IMG_SIZE, cfg.DATASET.IMG_SIZE)),
    T.Normalize(mean=cfg.DATASET.MEAN, std=cfg.DATASET.STD)
])
if args.dataset == 'pfpascal':
    dataset = PFPascalDataset(cfg, args.split, args.category)
    img_dataset = PFPascalImageDataset(cfg, args.split, args.category, transforms)
elif args.dataset == 'spair':
    dataset = SPairDataset(cfg, args.split, args.category)
    img_dataset = SPairImageDataset(cfg, args.split, args.category, transforms)
elif args.dataset == 'pfwillow':
    dataset = PFWillowDataset(cfg, 'test', args.category)
    img_dataset = PFWillowImageDataset(cfg, args.split, args.category, transforms)

loader = DataLoader(dataset, batch_size=args.batch_size, num_workers=args.num_workers, drop_last=False, shuffle=True)
img_loader = DataLoader(img_dataset, batch_size=1, num_workers=args.num_workers, shuffle=False)

evaluator = PCKEvaluator(cfg)

print("\n")
print(f"Dataset:{args.dataset}, Split:{args.split}, Category:{args.category}, Image size:{cfg.DATASET.IMG_SIZE}")
print(f"SD:{cfg.STABLE_DIFFUSION.VERSION}, Method:{cfg.FEATURE_EXTRACTOR.METHOD}, Prompt:{cfg.FEATURE_EXTRACTOR.PROMPT_TYPE}") 
print(f"Timestep:{cfg.FEATURE_EXTRACTOR.SELECT_TIMESTEP}, Layer:{cfg.FEATURE_EXTRACTOR.SELECT_LAYER}, Ensemble:{cfg.FEATURE_EXTRACTOR.ENSEMBLE_SIZE}")
print(f"FuseDino:{cfg.FEATURE_EXTRACTOR.FUSE_DINO}, Enable L2 Norm:{cfg.FEATURE_EXTRACTOR.ENABLE_L2_NORM}")
print(f"Evaluator: by {cfg.EVALUATOR.BY} (point/image)")

with torch.no_grad():

    if not ("CPM" in args.prompt_type and "Pair" in args.prompt_type):
        # a faster way to evaluate the matching. We firstly cache all feature maps and then do matching
        # cache all feature map
        featmap_dict = {}
        print("Prompt only depend on individual images, so we are caching all featmaps first...")
        for idx, batch in enumerate(tqdm(img_loader)):

            move_batch_to(batch, "cuda")

            imname = batch['identifier'][0]
            if "CPM" in args.prompt_type and "Img" in args.prompt_type:
                prompt = captioner(identifiers=[imname])
            else:
                prompt = prompter(imnames=[imname.split("-")[1]], class_names=[imname.split("-")[0]])
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                featmap = feature_extractor(image=batch['pixel_values'], prompt=prompt)
            
            featmap_dict[imname] = featmap.float()

    # do the real matching
    print("Do the real matching...")
    for idx, batch in enumerate(tqdm(loader)):
        
        move_batch_to(batch, "cuda")

        if ("CPM" in args.prompt_type and "Pair" in args.prompt_type):
            prompt = captioner(src_identifiers=batch["src_identifier"], trg_identifiers=batch["trg_identifier"])
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                featmaps0 = feature_extractor(image=batch['src_img'], prompt=prompt).float()
                featmaps1 = feature_extractor(image=batch['trg_img'], prompt=prompt).float()
        else:
            featmaps0 = torch.cat([featmap_dict[imname] for imname in batch['src_identifier']], dim=0)
            featmaps1 = torch.cat([featmap_dict[imname] for imname in batch['trg_identifier']], dim=0)

        batch['src_featmaps'] = featmaps0
        batch['trg_featmaps'] = featmaps1

        evaluator.evaluate_feature_map(batch, enable_l2_norm=cfg.FEATURE_EXTRACTOR.ENABLE_L2_NORM)

    evaluator.print_summarize_result()
    if args.category == 'all':
        file_name =  f"{args.dataset}_result_{args.prompt_type}_{args.category}_"+\
                     f"Layer{cfg.FEATURE_EXTRACTOR.SELECT_LAYER}-"+\
                     f"Timestep{cfg.FEATURE_EXTRACTOR.SELECT_TIMESTEP}-"+\
                     f"L2Norm{cfg.FEATURE_EXTRACTOR.ENABLE_L2_NORM}-"+\
                     f"FuseDino{cfg.FEATURE_EXTRACTOR.FUSE_DINO}.txt"
        save_file = os.path.join(cfg.FEATURE_EXTRACTOR.PROMPT_CACHE_ROOT, cfg.FEATURE_EXTRACTOR.PROMPT_TYPE, file_name)
        os.makedirs(os.path.join(cfg.FEATURE_EXTRACTOR.PROMPT_CACHE_ROOT, cfg.FEATURE_EXTRACTOR.PROMPT_TYPE), exist_ok=True)
        evaluator.save_result(save_file)
