import torch
from torch.utils.data import DataLoader
from torch.optim import Adam
from torch.utils.tensorboard import SummaryWriter
import torchvision.transforms as T

import os
import argparse
import numpy as np
from tqdm import tqdm
from datetime import timedelta

from dataset import SPairDataset, SPairImageDataset, PFPascalDataset, PFPascalImageDataset
from config import get_default_defaults
from utils.misc import str2bool, move_batch_to
from src.stable_diffusion.sd_feature_extractor import SDFeatureExtraction
from src.stable_diffusion.hybrid_captioner import HybridCaptioner
from src.loss import GaussianCrossEntropyLoss
from utils.evaluator import PCKEvaluator

from accelerate import Accelerator
from accelerate.utils import InitProcessGroupKwargs
from diffusers.optimization import get_scheduler

parser = argparse.ArgumentParser()
parser.add_argument('--config_file', default='config/learnedToken.py', type=str)

parser.add_argument('--dataset', default='spair', type=str)
parser.add_argument('--captioner_config', type=str, default="Pair-DINO-Feat-G25-C50", help='[Img|Pair]-[CLIP|DINO]-[Head|Feat]-G[int]-C[int]')

parser.add_argument('--batch_size', default=1, type=int)
parser.add_argument('--epochs', default=50, type=int)
parser.add_argument('--init_lr', default=0.1, type=float)
parser.add_argument('--end_lr', default=0.01, type=float)
parser.add_argument("--scheduler", type=str, default="constant",help='Choose between ["linear", "constant", "piecewise_constant"]')
parser.add_argument("--scheduler_power", type=float, default=1.0)
parser.add_argument("--scheduler_step_rules", type=str, default=None)
parser.add_argument('--num_workers', default=2, type=int)

args = parser.parse_args()

np.random.seed(0)
torch.manual_seed(0)

cfg = get_default_defaults()
cfg.merge_from_file(args.config_file)

# override dataset name in config file
cfg.DATASET.NAME = args.dataset

prompt_type = f"CPM_{args.dataset}_sd{cfg.STABLE_DIFFUSION.VERSION}_{args.captioner_config}"

# override prompt type and ensemble size
cfg.FEATURE_EXTRACTOR.PROMPT_TYPE = prompt_type
cfg.FEATURE_EXTRACTOR.ENSEMBLE_SIZE = 1

logging_dir = os.path.join(cfg.FEATURE_EXTRACTOR.LOG_ROOT, args.dataset, f"{prompt_type}_{args.scheduler}_lr{args.init_lr}")
output_dir = os.path.join(cfg.FEATURE_EXTRACTOR.PROMPT_CACHE_ROOT, f"{prompt_type}")

# create accelerator
kwargs = InitProcessGroupKwargs(timeout=timedelta(seconds=7200))
accelerator = Accelerator(kwargs_handlers=[kwargs])

# create dataset
if args.dataset == "spair":
    train_dataset = SPairDataset(cfg, split="trn", category="all")
    val_dataset = SPairDataset(cfg, split="val", category="all")
    transforms = T.Compose([
            T.ToTensor(),
            T.Resize((cfg.DATASET.IMG_SIZE, cfg.DATASET.IMG_SIZE)),
            T.Normalize(mean=cfg.DATASET.MEAN, std=cfg.DATASET.STD)
        ])
    img_dataset = SPairImageDataset(cfg, "val", "all", transforms)
elif args.dataset == "pfpascal":
    train_dataset = PFPascalDataset(cfg, split="trn", category="all")
    val_dataset = PFPascalDataset(cfg, split="val", category="all")
    transforms = T.Compose([
            T.ToTensor(),
            T.Resize((cfg.DATASET.IMG_SIZE, cfg.DATASET.IMG_SIZE)),
            T.Normalize(mean=cfg.DATASET.MEAN, std=cfg.DATASET.STD)
        ])
    img_dataset = PFPascalImageDataset(cfg, "val", "all", transforms)

# create dataloader
train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
img_loader = DataLoader(img_dataset, batch_size=1, shuffle=False)

# create model
feature_extractor = SDFeatureExtraction(cfg)
for param in feature_extractor.parameters():
    param.requires_grad = False
feature_extractor.to(accelerator.device)

# create hybrid captioner
captioner = HybridCaptioner(prompt_type, cfg.FEATURE_EXTRACTOR.ASSET_ROOT, accelerator.device)
for param in captioner.text_model_embeddings.parameters():
    param.requires_grad = False
captioner.to(accelerator.device)

# create optimizer
learned_param = []
for name, param in captioner.named_parameters():
    net_param = []
    explicit_param = []
    if "text_model_embeddings" not in name:     # we don't change text_model_embeddings
        if "linear" in name:
            net_param.append(param)
        else:
            explicit_param.append(param)
    learned_param.append({"params": explicit_param})
    learned_param.append({"params": net_param, "lr": 0.1*args.init_lr})
optimizer = Adam(learned_param, lr=args.init_lr)

# create scheduler
lr_scheduler = get_scheduler(
    args.scheduler,
    optimizer=optimizer,
    num_training_steps=args.epochs,
    num_warmup_steps = 0,
    step_rules=args.scheduler_step_rules,
    power=args.scheduler_power
)

# wrap everything using accelerator
feature_extractor, captioner, train_loader, optimizer, lr_scheduler = \
    accelerator.prepare(feature_extractor, captioner, train_loader, optimizer, lr_scheduler)

if os.path.exists(logging_dir):
    print("Found saved state, continue training...")
    accelerator.load_state(logging_dir)
    start_epoch = torch.load(os.path.join(logging_dir, "epoch.pt")) + 1
else:
    start_epoch = 0

accelerator.wait_for_everyone()

# create wrtier
if accelerator.is_main_process:
    writer = SummaryWriter(log_dir=logging_dir)

loss_fn = GaussianCrossEntropyLoss()

evaluator = PCKEvaluator(cfg)
best_pck = 0
progress_bar_epoch = tqdm(range(start_epoch, args.epochs), disable=not accelerator.is_main_process)
for epoch in range(start_epoch, args.epochs):

    evaluator.clear_result()

    progress_bar = tqdm(range(len(train_loader)), disable=not accelerator.is_main_process)
    for idx, batch in enumerate(train_loader):

        optimizer.zero_grad()

        if "Img" in prompt_type:
            src_prompts = captioner.module(identifiers=batch['src_identifier'])
            trg_prompts = captioner.module(identifiers=batch['trg_identifier'])
        else:
            src_prompts = captioner.module(src_identifiers=batch['src_identifier'], trg_identifiers=batch['trg_identifier'])
            trg_prompts = src_prompts

        src_featmaps = feature_extractor(batch['src_img'], src_prompts)
        trg_featmaps = feature_extractor(batch['trg_img'], trg_prompts)

        lossfn_input = {
            'src_featmaps': src_featmaps,
            'trg_featmaps': trg_featmaps,
            'src_kps': batch['src_kps'],
            'trg_kps': batch['trg_kps'],
            'src_imgsize': batch['src_img'].shape[2:],
            'trg_imgsize': batch['trg_img'].shape[2:],
            'npts': batch['n_pts'],
            'category': batch['category'],
            'softmax_temp': 0.04,
            'enable_l2_norm': cfg.FEATURE_EXTRACTOR.ENABLE_L2_NORM
        }

        loss = loss_fn(**lossfn_input)

        accelerator.backward(loss)
        
        log_loss = accelerator.gather(loss)
        log_loss = log_loss.mean().item()

        if accelerator.is_main_process:
            writer.add_scalar("train_loss", log_loss, epoch*len(train_loader)+idx)
            writer.add_scalar("lr", optimizer.param_groups[0]['lr'], epoch*len(train_loader)+idx)

        optimizer.step()
        progress_bar.update(1)
        progress_bar.set_postfix(loss=log_loss, lr=optimizer.param_groups[0]['lr'])

    lr_scheduler.step()

    accelerator.wait_for_everyone()

    # validation (since we are overfitting, it is the same as the training data)
    with torch.no_grad():
        if accelerator.is_main_process:
            if not "Pair" in prompt_type:
                # a faster way to evaluate the matching. We firstly cache all feature maps and then do matching
                # cache all feature map
                featmap_dict = {}
                print("Prompt only depend on individual images, so we are caching all featmaps first...")
                for idx, batch in enumerate(tqdm(img_loader)):

                    move_batch_to(batch, "cuda")

                    imname = batch['identifier'][0]
                    prompt = captioner.module(identifiers=[imname])

                    with torch.autocast(device_type="cuda", dtype=torch.float16):
                        featmap = feature_extractor(image=batch['pixel_values'], prompt=prompt)
                    
                    featmap_dict[imname] = featmap.float()


            # do the real matching
            print("Do the real matching...")
            for idx, batch in enumerate(tqdm(val_loader)):
                
                move_batch_to(batch, "cuda")

                if "Pair" in prompt_type:
                    prompt = captioner.module(src_identifiers=batch["src_identifier"], trg_identifiers=batch["trg_identifier"])
                    with torch.autocast(device_type="cuda", dtype=torch.float16):
                        featmaps0 = feature_extractor(image=batch['src_img'], prompt=prompt)
                        featmaps1 = feature_extractor(image=batch['trg_img'], prompt=prompt)
                else:
                    featmaps0 = torch.cat([featmap_dict[imname] for imname in batch['src_identifier']], dim=0)
                    featmaps1 = torch.cat([featmap_dict[imname] for imname in batch['trg_identifier']], dim=0)
                
                batch['src_featmaps'] = featmaps0
                batch['trg_featmaps'] = featmaps1

                evaluator.evaluate_feature_map(batch, enable_l2_norm=cfg.FEATURE_EXTRACTOR.ENABLE_L2_NORM)

            pck = np.array(evaluator.result["nn_pck0.1"]["all"]).mean()
            writer.add_scalar("val_acc", pck, epoch)

            if pck > best_pck:
                best_pck = pck
                os.makedirs(output_dir, exist_ok=True)
                captioner.module.save_state_dict(os.path.join(output_dir, "ckpt.pt"))

            if not "Pair" in prompt_type:

                del featmap_dict

            # save current state
            accelerator.save_state(logging_dir)
            torch.save(epoch, os.path.join(logging_dir, "epoch.pt"))

    progress_bar_epoch.update(1)
    progress_bar_epoch.set_postfix(epoch=epoch)
    accelerator.wait_for_everyone()

    torch.cuda.empty_cache()