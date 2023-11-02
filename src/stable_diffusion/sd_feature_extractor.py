from typing import Union, List

import torch
import torch.nn as nn
import torchvision.transforms.functional as TF

from .custom_stable_diffusion_pipeline import CustomStableDiffusionPipeline
from transformers import logging, AutoModel

from utils.misc import *
from utils.matching import L2normalization
from .custom_unet_2d_condition import UNet2DConditionModel

class SDFeatureExtraction(nn.Module):

    SD_VERSIONS = {
        "1-3": "CompVis/stable-diffusion-v1-3",
        "1-4": "CompVis/stable-diffusion-v1-4",
        "1-5": "runwayml/stable-diffusion-v1-5",
        "2-1": "stabilityai/stable-diffusion-2-1",
    }

    def __init__(self, cfg):

        super().__init__()
        self.cfg = cfg

        # suppress hugging face logging other than error 
        logging.set_verbosity_error()

        # (1) select stable diffusion version
        sd_id = self.SD_VERSIONS[cfg.STABLE_DIFFUSION.VERSION]
        
        # (2.1) construct the customed unet and load weight
        self.unet = UNet2DConditionModel.from_pretrained(sd_id, subfolder="unet")
        
        # (2.2) construct the diffusion pipeline
        pipe = CustomStableDiffusionPipeline.from_pretrained(sd_id, unet=self.unet)
        if cfg.STABLE_DIFFUSION.SAVE_MEMORY:
            pipe.enable_attention_slicing()
            pipe.enable_xformers_memory_efficient_attention()

        # (3) get scheduler and vae
        self.scheduler = pipe.scheduler
        self.vae = pipe.vae
        
        # (4) get tokenizer and text_encoder if use "text" or "blip"
        self.text_encoder = pipe.text_encoder           # register text_encoder to model so it will move to current device when using _encode_prompt
        self.tokenizer = pipe.tokenizer
        self.explicit_captioner = pipe._encode_prompt


        # (5) get ensemble
        self.ensb = cfg.FEATURE_EXTRACTOR.ENSEMBLE_SIZE

        # (6) fuse dino if chosen to
        if cfg.FEATURE_EXTRACTOR.FUSE_DINO:
            self.dino = AutoModel.from_pretrained('facebook/dinov2-base')
        
    def forward(self, 
                image: torch.Tensor, 
                prompt: Union[List[str], torch.Tensor]):
        """
        IN:
            image : torch.Tensor
                A batch of images to be encoded, shape (B,C,H,W).
            prompt : Union[List[str], torch.Tensor])
                A list of text prompts or a batch of text embedding corresponding to the input images.
        OUT:
            torch.Tensor, a batch of feature maps with shape (B,c,h,w).
        """
        
        _device = image.device

        # (1) get image embedding
        img_embed = self.vae.encode(image).latent_dist.sample()
        img_embed = img_embed * self.vae.config.scaling_factor

        # (2.1) get prompt embedding with text tokenizer and text encoder if use "text" or "blip"
        if "text" in self.cfg.FEATURE_EXTRACTOR.PROMPT_TYPE and isinstance(prompt, list):
            prompt_embed = self.explicit_captioner(prompt=prompt, 
                                                   device=_device, 
                                                   num_images_per_prompt=1, 
                                                   do_classifier_free_guidance=False)

        # (2.2) use prompt argument as the prompt embedding if use "sd2-1_learned_hidden_state"
        elif "HiddenState" in self.cfg.FEATURE_EXTRACTOR.PROMPT_TYPE and isinstance(prompt, torch.Tensor):
            prompt_embed = self.explicit_captioner(prompt=None,
                                                   prompt_embeds=prompt,
                                                   device=_device, 
                                                   num_images_per_prompt=1, 
                                                   do_classifier_free_guidance=False)

        # (2.3) use prompt argument as the prompt embedding if use "sd2-1_learned_hidden_state"
        elif ('TokenEmbeds' in self.cfg.FEATURE_EXTRACTOR.PROMPT_TYPE or "CPM" in self.cfg.FEATURE_EXTRACTOR.PROMPT_TYPE) \
            and isinstance(prompt, torch.Tensor):
            prompt_embed = self.explicit_captioner(prompt=None,
                                                   token_embeds=prompt,
                                                   device=_device,
                                                   num_images_per_prompt=1,
                                                   do_classifier_free_guidance=False)

        # (3) ensemble
        if self.ensb > 1:
            img_embed = img_embed.repeat_interleave(repeats=self.ensb, dim=0) 
            prompt_embed = prompt_embed.repeat_interleave(repeats=self.ensb, dim=0) 

        # (4) add noise 
        timestep = self.cfg.FEATURE_EXTRACTOR.SELECT_TIMESTEP
        timestep = torch.tensor(timestep, dtype=torch.long, device=_device)
        noise = torch.randn_like(img_embed)
        noisy_img_embed = self.scheduler.add_noise(
            original_samples=img_embed, 
            noise=noise, 
            timesteps=timestep)
        
        # (5) reverse process
        featmap = self.unet(
            sample=noisy_img_embed, 
            timestep=timestep, 
            encoder_hidden_states=prompt_embed, 
            feature_extractor=self.cfg.FEATURE_EXTRACTOR.METHOD)['featmaps']
        
        # (6) select used layer
        featmap = featmap[self.cfg.FEATURE_EXTRACTOR.SELECT_LAYER]

        # (7) average over multiple ensembles
        if self.ensb > 1:
            featmap = featmap.view(-1, self.ensb, *featmap.shape[1:]).mean(dim=1)

        # (8) fuse with dino if chosen to
        if self.cfg.FEATURE_EXTRACTOR.FUSE_DINO:
            b, _, h, w = featmap.shape
            img = TF.resize(image, (14*h, 14*w))
            dino_featmap = self.dino(img).last_hidden_state[:, 1:, :].permute(0, 2, 1).view(b, -1, h, w)
            featmap = L2normalization(featmap, dim=1)
            dino_featmap = L2normalization(dino_featmap, dim=1)
            featmap = torch.cat([0.5*featmap, 0.5*dino_featmap], dim=1)            

        return featmap 