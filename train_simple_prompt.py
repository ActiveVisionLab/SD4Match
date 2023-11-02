import torch
from torch.utils.data import DataLoader
from torch.optim import Adam
import torch.distributed as dist
from torch.utils.tensorboard import SummaryWriter
import torchvision.transforms as T

import os
import argparse
import numpy as np
from tqdm import tqdm

from dataset import SPairDataset, SPairImageDataset, PFPascalDataset, PFPascalImageDataset
from config import get_default_defaults
from utils.misc import str2bool, move_batch_to
from src.stable_diffusion.sd_feature_extractor import SDFeatureExtraction
from src.stable_diffusion.hidden_state import HiddenStateHandler, ReferenceDictGenerator
from src.stable_diffusion.prompt import PromptManager
from src.loss import GaussianCrossEntropyLoss
from utils.evaluator import PCKEvaluator

from accelerate import Accelerator
from diffusers.optimization import get_scheduler

parser = argparse.ArgumentParser()
parser.add_argument('--config_file', default='config/learnedToken.py', type=str)

parser.add_argument('--dataset', default='spair', type=str)

parser.add_argument('--prompt_option', type=str, default="class", help='choose between [single, class]')
parser.add_argument('--learnable_seq_length', type=int, required=True)
parser.add_argument('--learn_hidden_state', type=str2bool, default=False)

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

# determine output directory
if args.learn_hidden_state:
    prompt_type = "HiddenState"
else:
    prompt_type = f"TokenEmbeds{args.learnable_seq_length}"

prompt_type = f"{args.dataset}_sd{cfg.STABLE_DIFFUSION.VERSION}_{prompt_type}"

if args.prompt_option.lower() == 'single':
    prompt_type = "Single_"+prompt_type
elif args.prompt_option.lower() == 'class':
    prompt_type = "Class_"+prompt_type
else:
    raise ValueError(f"unsupported caption option {args.prompt_option}")

# override prompt type and ensemble size
cfg.FEATURE_EXTRACTOR.PROMPT_TYPE = prompt_type
cfg.FEATURE_EXTRACTOR.ENSEMBLE_SIZE = 1

logging_dir = os.path.join(cfg.FEATURE_EXTRACTOR.LOG_ROOT, args.dataset,
                           f"{prompt_type}_{args.scheduler}_lr{args.init_lr}")
output_dir = os.path.join(cfg.FEATURE_EXTRACTOR.PROMPT_CACHE_ROOT, prompt_type)

# create accelerator
accelerator = Accelerator()

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

# create hidden state handler
# get image set first
prompt_names = list(set(train_dataset.src_identifiers + train_dataset.trg_identifiers))

# only create ref_dict_generator on rank 0 and broadcast it to other machines to make sure all processes have the same ref_dict
if dist.get_rank() == 0:
    ref_dict_generator = ReferenceDictGenerator(
        sd_version=cfg.STABLE_DIFFUSION.VERSION,
        caption_option=args.prompt_option,
        device = accelerator.device
    )
    ref_dict = [ref_dict_generator(prompt_names)]

    del ref_dict_generator
else:
    ref_dict = [None]

accelerator.wait_for_everyone()

dist.broadcast_object_list(ref_dict, src=0, device=accelerator.device)
ref_dict = ref_dict[0]

torch.cuda.empty_cache()
accelerator.wait_for_everyone()

prompt_handler = HiddenStateHandler(
    reference_dict=ref_dict,
    learnable_length=args.learnable_seq_length,
    learn_hidden_state=args.learn_hidden_state,
    tokenizer=feature_extractor.tokenizer,
    text_encoder=feature_extractor.text_encoder,
    device=accelerator.device
)

# create optimizer
optimizer = Adam(prompt_handler.parameters(), lr=args.init_lr)

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
feature_extractor, prompt_handler, train_loader, optimizer, lr_scheduler = \
    accelerator.prepare(feature_extractor, prompt_handler, train_loader, optimizer, lr_scheduler)

# create wrtier
if accelerator.is_main_process:
    writer = SummaryWriter(log_dir=logging_dir)

loss_fn = GaussianCrossEntropyLoss()

evaluator = PCKEvaluator(cfg)
eval_prompter = PromptManager(cfg)
best_pck = 0
progress_bar_epoch = tqdm(range(args.epochs), disable=not accelerator.is_main_process)
for epoch in range(args.epochs):

    evaluator.clear_result()

    progress_bar = tqdm(range(len(train_loader)), disable=not accelerator.is_main_process)
    for idx, batch in enumerate(train_loader):

        optimizer.zero_grad()

        src_prompts = prompt_handler.module.get(batch['src_identifier'])
        trg_prompts = prompt_handler.module.get(batch['trg_identifier'])

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

            eval_prompter.saved_prompt = prompt_handler.module._produce_saving_dict()

            featmap_dict = {}
            ## cache all featmaps
            print("Caching all featmaps...")
            for idx, batch in enumerate(tqdm(img_loader)):
                move_batch_to(batch, accelerator.device)
                imname = batch['identifier'][0]

                prompt = eval_prompter(imnames=[imname.split("-")[1]], class_names=[imname.split("-")[0]])

                with torch.autocast(device_type="cuda", dtype=torch.float16):
                    featmap = feature_extractor(image=batch['pixel_values'], prompt=prompt)

                featmap_dict[imname] = featmap

            ## do the real matching
            print("Do the real matching...")
            for idx, batch in enumerate(tqdm(val_loader)):
                
                move_batch_to(batch, accelerator.device)

                src_imnames = [f"{batch['category'][i]}-{batch['src_imname'][i][:-4]}" for i in range(len(batch['src_imname']))]
                trg_imnames = [f"{batch['category'][i]}-{batch['trg_imname'][i][:-4]}" for i in range(len(batch['trg_imname']))]

                featmaps0 = torch.cat([featmap_dict[imname] for imname in src_imnames], dim=0)
                featmaps1 = torch.cat([featmap_dict[imname] for imname in trg_imnames], dim=0)
                
                batch['src_featmaps'] = featmaps0
                batch['trg_featmaps'] = featmaps1

                evaluator.evaluate_feature_map(batch, enable_l2_norm=cfg.FEATURE_EXTRACTOR.ENABLE_L2_NORM)

            del featmap_dict

            pck = np.array(evaluator.result["nn_pck0.1"]["all"]).mean()
            writer.add_scalar("val_acc", pck, epoch)

            if pck > best_pck:
                best_pck = pck
                prompt_handler.module.save_learnable_params(output_dir)

    progress_bar_epoch.update(1)
    progress_bar_epoch.set_postfix(epoch=epoch)
    accelerator.wait_for_everyone()

    torch.cuda.empty_cache()