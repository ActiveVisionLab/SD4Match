import torch
import torch.nn as nn

import os
from tqdm import tqdm
from typing import List, Dict

from transformers import CLIPTextModel, CLIPTokenizer

import numpy as np


TEXT_TEMPLATE = "a photo of a {category}"

class HiddenStateHandler(nn.Module):

    def __init__(self, 
                 reference_dict:Dict,
                 learnable_length:int,
                 learn_hidden_state:bool=True,
                 tokenizer: CLIPTokenizer=None,
                 text_encoder:CLIPTextModel=None,
                 device="cpu"
                 ):
        """
        IN:
            reference_dict: 
                A dictionary that contain the conversion from image name to prompt
            learnable_length: 
                The length of the sequence that is learnable. Only affact when we learn token embedding
            learn_hidden_state: 
                If set to True, we directly learn the entire text hidden state. Otherwise, we learn the token_embedding.
            tokenizer: 
                The tokenizer to convert text into token.
            text_encoder: 
                The text_encoder to extract hidden state of the text template.
            device:
                device of the parameter.
        """
        super(HiddenStateHandler, self).__init__()
        
        '''Current implementation is based on CLIPTokenizer. So check whether tokenizer is a clip tokenizer or not'''
        assert isinstance(tokenizer, CLIPTokenizer), "Tokenizer is not a CLIPTokenizer."

        self.ref_dict = reference_dict
        self.learn_hidden_state = learn_hidden_state
        self.max_seq_length = tokenizer.model_max_length
        self.hidden_state_dim = text_encoder.config.hidden_size
        self.device = device

        self.tokenizer = tokenizer
        self.text_encoder = text_encoder
        self.learnable_parameters = nn.ParameterDict()
        
        print("Initialize Learnable Parameters")
        progress_bar = tqdm(range(len(self.ref_dict["imnames_to_param_ids"])))
        for imname, param_ids in self.ref_dict["imnames_to_param_ids"].items():
            if param_ids not in self.learnable_parameters:
                if learn_hidden_state:
                    self.learnable_parameters.update({param_ids: nn.Parameter(torch.rand(1, self.max_seq_length, self.hidden_state_dim), requires_grad=True)})
                else:
                    ids = tokenizer("", truncation=True, return_tensors="pt")["input_ids"].to("cuda")
                    fixed_token_length = ids.shape[-1]
                    actual_learnable_length = min(self.max_seq_length-fixed_token_length, learnable_length)
                    rand_ids = torch.randint(0, tokenizer.vocab_size, (1, actual_learnable_length), device=device)
                    ids = torch.cat([ids[:, :-1], rand_ids, ids[:, -1:]], dim=-1)
                    with torch.no_grad():
                        token_embed = text_encoder.text_model.embeddings(input_ids=ids)
                    self.learnable_parameters.update({param_ids: nn.Parameter(token_embed[:, fixed_token_length-1:-1], requires_grad=True)})
            else:
                pass

            progress_bar.update(1)

        self.learnable_parameters.to(device)


    def get(self, imnames: List[str]):
        
        output = []

        for imname in imnames:
                
            output.append(self.query_learnable_parameters(self.ref_dict["imnames_to_param_ids"][imname]))

        return torch.cat(output, dim=0)


    def query_learnable_parameters(self, key):
        if self.learn_hidden_state:
            return self.learnable_parameters[key]
        else:
            return self._produce_token_embeds("", self.learnable_parameters[key])


    def _produce_token_embeds(self, text, learnable_param):
        """Produce the final token_embeds using text template and learnable_params"""
        ids = self.tokenizer(text, truncation=True, return_tensors="pt")["input_ids"].to("cuda")
                
        fixed_token_length = ids.shape[-1]
        actual_learnable_length = learnable_param.shape[1]
        padding_length = self.max_seq_length - fixed_token_length - actual_learnable_length
        
        dummy_ids = torch.randint(0, self.tokenizer.vocab_size, (1, actual_learnable_length), device=self.device)
        padding_ids = torch.zeros(1, padding_length, dtype=torch.int64, device=self.device)
        
        ids = torch.cat([ids[:, :-1], dummy_ids, ids[:, -1:], padding_ids], dim=-1)
        
        with torch.no_grad():
            token_embed = self.text_encoder.text_model.embeddings(input_ids=ids)

        token_embed = torch.cat([token_embed[:, :fixed_token_length-1], learnable_param, 
                                    token_embed[:, fixed_token_length-1+actual_learnable_length:]], dim=1)

        return token_embed


    def _produce_saving_dict(self):

        out = {}

        caption_option = self.ref_dict["caption_option"]

        if caption_option == "class":
            for k, v in self.ref_dict["imnames_to_param_ids"].items():
                category = k.split("-")[0]
                if category not in out:
                    out.update({category: self.query_learnable_parameters(v)})
                else:
                    pass

        elif caption_option == "single":
            out.update({"dataset": self.query_learnable_parameters("0")})

        return out
    

    def save_learnable_params(self, root_dir: str):

        print(f"Save all learned parameters to {root_dir}")
        
        out = self._produce_saving_dict()

        os.makedirs(root_dir, exist_ok=True)

        torch.save(out, os.path.join(root_dir, "caption.pt"))

        print("Done!")



class ReferenceDictGenerator:

    def __init__(self,
                 sd_version:str,
                 caption_option:str,
                 device):

        self.sd_version = sd_version
        self.caption_option = caption_option
        self.device = device


    @ torch.no_grad()
    def __call__(self, imnames:List[str]):

        ref_dict = {}
        ref_dict.update({"imnames_to_param_ids": {}})
        ref_dict.update({"caption_option": self.caption_option})

        category_list = list(set([imname.split("-")[0] for imname in imnames]))
        category_list.sort()

        if self.caption_option == 'class':
            for idx, imname in enumerate(imnames):
                ref_dict["imnames_to_param_ids"].update({imname: str(category_list.index(imname.split("-")[0]))})
        elif self.caption_option == 'single':
            for idx, imname in enumerate(imnames):
                ref_dict["imnames_to_param_ids"].update({imname: str(0)})

        return ref_dict