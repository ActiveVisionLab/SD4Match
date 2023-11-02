import os
import torch
import torch.nn as nn

from transformers import CLIPTokenizer, CLIPTextModel

from typing import Any, List, Mapping



class HybridCaptioner(nn.Module):

    def __init__(self, name, feat_cache_root, device):
        super().__init__()
        _, dataset, sd_version, config = name.split("_")
        mode, feature_extractor, feature_type, global_length, cond_length = config.split("-")
        
        self.max_learnable_length = 75
        self.mode = mode
        self.feature_extractor = feature_extractor
        self.feature_type = feature_type
        self.global_length = int(global_length[1:])
        self.cond_length = int(cond_length[1:])

        self.device = device

        out_dim = 1024

        self.check_config()

        # build everything according to config
        # load cached feature
        if self.feature_extractor == "CLIP":
            self.feat_cache_dict = torch.load(os.path.join(feat_cache_root, 'CLIP-ViT-H', dataset, 'cached_output.pt'), map_location=device)
            if self.feature_type == "Head":
                in_dim = 1024
            elif self.feature_type == "Feat":
                in_dim = 1280
                n_patch = 256
        elif self.feature_extractor == "DINO":
            self.feat_cache_dict = torch.load(os.path.join(feat_cache_root, 'DINOv2', dataset, 'cached_output.pt'), map_location=device)
            in_dim = 768
            n_patch = 256

        text_encoder = CLIPTextModel.from_pretrained("stabilityai/stable-diffusion-2-1", subfolder="text_encoder")
        self.text_model_embeddings = text_encoder.text_model.embeddings
        self.tokenizer = CLIPTokenizer.from_pretrained("stabilityai/stable-diffusion-2-1", subfolder="tokenizer")

        # construct global embeds
        if self.global_length > 0:
            self.global_embeds = nn.Parameter(torch.rand(1, self.global_length, out_dim))

        # construct network that produces cond seq
        if self.mode == "Img":
            if self.feature_type == "Head":
                self.positional_embedding = nn.Parameter(torch.rand(1, self.cond_length, out_dim))
                self.linear = nn.Linear(in_dim, out_dim, bias=True)
                self.alpha_cond = nn.Parameter(torch.rand(1, self.cond_length, out_dim))
            
            elif self.feature_type == "Feat":
                self.positional_embedding = nn.Parameter(torch.rand(1, self.cond_length, out_dim))
                self.linear = nn.Linear(in_dim, out_dim, bias=True)
                self.adaptive_pool = nn.AdaptiveMaxPool1d(self.cond_length)
                self.alpha_cond = nn.Parameter(torch.rand(1, self.cond_length, out_dim))

        elif self.mode == "Pair":
            if self.feature_type == "Head":
                self.positional_embedding = nn.Parameter(torch.rand(1, self.cond_length, out_dim))
                self.linear = nn.Linear(in_dim*2, out_dim, bias=True)
                self.alpha_cond = nn.Parameter(torch.rand(1, self.cond_length, out_dim))

            elif self.feature_type == "Feat":
                self.positional_embedding = nn.Parameter(torch.rand(1, self.cond_length, out_dim))
                self.linear_C = nn.Linear(in_dim*2, out_dim, bias=True)
                self.linear_N = nn.Linear(n_patch, n_patch, bias=True)
                self.adaptive_pool = nn.AdaptiveMaxPool1d(self.cond_length)
                self.alpha_cond = nn.Parameter(torch.rand(1, self.cond_length, out_dim))


    def check_config(self):

        if self.mode not in ["Img", "Pair"]:
            raise ValueError(f"Unsupported mode {self.mode} in Hybrid Captioner. Choose between ['Img', 'Pair'].")
        
        if self.feature_extractor not in ["CLIP", "DINO"]:
            raise ValueError(f"Unsupported feature extractor: {self.feature_extractor} in Hybrid Captioner. Choose between ['CLIP', 'DINO'].")
        
        if self.feature_type not in ["Feat", "Head"]:
            raise ValueError(f"Unsupported feature type: {self.feature_type} in Hybrid Captioner. Choose between ['Feat', 'Head'].")
        
        if self.global_length > self.max_learnable_length:
            raise ValueError(f"Global length {self.global_length} is longer than max length {self.max_learnable_length}.")
        
        if self.cond_length > self.max_learnable_length:
            raise ValueError(f"Conditional length {self.cond_length} is longer than max length {self.max_learnable_length}.")
        
        if self.global_length + self.cond_length > self.max_learnable_length:
            raise ValueError(f"The sum ({self.global_length+self.cond_length}) of global length ({self.global_length})" + 
                             f"and conditional length ({self.cond_length}) is longer than learnable max length ({self.max_learnable_length}).")


    def forward(self,
                identifiers: List[str] = None, 
                src_identifiers: List[str] = None, 
                trg_identifiers: List[str] = None):

        # check inputs
        if self.mode == "Img":
            assert identifiers is not None, "Mode is Img, but identifiers is None."
        elif self.mode == "Pair":
            assert (src_identifiers is not None) and (trg_identifiers is not None), "Mode is Pair, but one of src_identifiers and trg_identifiers being None."

        # get feat or src_feat and trg_feat according to config
        if self.mode == "Img":
            feat = []
            for iden in identifiers:
                if self.feature_type == "Head":
                    if self.feature_extractor == "CLIP":
                        feat.append(self.feat_cache_dict[iden].image_embeds)
                    elif self.feature_extractor == "DINO":
                        feat.append(self.feat_cache_dict[iden].pooler_output)
                elif self.feature_type == "Feat":
                    feat.append(self.feat_cache_dict[iden].last_hidden_state[:, 1:, :])
            feat = torch.cat(feat, dim=0)

        elif self.mode == "Pair":
            src_feat = []
            trg_feat = []
            for src_iden, trg_iden in zip(src_identifiers, trg_identifiers):
                if self.feature_type == "Head":
                    if self.feature_extractor == "CLIP":
                        src_feat.append(self.feat_cache_dict[src_iden].image_embeds)
                        trg_feat.append(self.feat_cache_dict[trg_iden].image_embeds)
                    elif self.feature_extractor == "DINO":
                        src_feat.append(self.feat_cache_dict[src_iden].pooler_output)
                        trg_feat.append(self.feat_cache_dict[trg_iden].pooler_output)
                elif self.feature_type == "Feat":
                    src_feat.append(self.feat_cache_dict[src_iden].last_hidden_state[:, 1:, :])
                    trg_feat.append(self.feat_cache_dict[trg_iden].last_hidden_state[:, 1:, :])
            src_feat = torch.cat(src_feat, dim=0)
            trg_feat = torch.cat(trg_feat, dim=0)

        # Generate conditional token embedding
        if self.mode == "Img":
            if self.feature_type == "Head":
                x = self.linear(feat).unsqueeze(1)
                x = self.positional_embedding + x
                cond_embeds = torch.tanh(self.alpha_cond) * x
            elif self.feature_type == "Feat":
                x = self.linear(feat)
                x = self.adaptive_pool(x.permute(0, 2, 1)).permute(0, 2, 1)
                x = self.positional_embedding + x
                cond_embeds = torch.tanh(self.alpha_cond) * x
                
        elif self.mode == "Pair":
            if self.feature_type == "Head":
                feat = torch.cat([src_feat, trg_feat], dim=1)
                x = self.linear(feat).unsqueeze(1)
                x = self.positional_embedding + x
                cond_embeds = torch.tanh(self.alpha_cond) * x
            elif self.feature_type == "Feat":
                feat = torch.cat([src_feat, trg_feat], dim=2)
                x = self.linear_C(feat)
                x = self.linear_N(x.permute(0, 2, 1)).permute(0, 2, 1)
                x = self.adaptive_pool(x.permute(0, 2, 1)).permute(0, 2, 1)
                x = self.positional_embedding + x
                cond_embeds = torch.tanh(self.alpha_cond) * x

        # put up everything
        ids = self.tokenizer("", truncation=True, return_tensors="pt")["input_ids"].to(self.device)
                
        fixed_token_length = ids.shape[-1]
        actual_learnable_length = self.global_length + self.cond_length
        padding_length = self.max_learnable_length + 2 - fixed_token_length - actual_learnable_length
        
        dummy_ids = torch.randint(0, self.tokenizer.vocab_size, (1, actual_learnable_length), device=self.device)
        padding_ids = torch.zeros(1, padding_length, dtype=torch.int64, device=self.device)
        
        ids = torch.cat([ids[:, :-1], dummy_ids, ids[:, -1:], padding_ids], dim=-1)

        with torch.no_grad():
            dummy_embed = self.text_model_embeddings(input_ids=ids)

        token_embed = []
        for i in range(cond_embeds.shape[0]):
            if self.global_length > 0:
                token_embed.append(torch.cat([dummy_embed[:, :fixed_token_length-1], self.global_embeds, cond_embeds[i:i+1],
                                            dummy_embed[:, fixed_token_length-1+actual_learnable_length:]], dim=1))
            else:
                token_embed.append(torch.cat([dummy_embed[:, :fixed_token_length-1], cond_embeds[i:i+1],
                                            dummy_embed[:, fixed_token_length-1+actual_learnable_length:]], dim=1))
        token_embed = torch.cat(token_embed, dim=0)

        return token_embed
    

    def save_state_dict(self, file_name):

        state_dict = self.state_dict()

        # we do not save the weight of text model embeddings
        state_dict.pop('text_model_embeddings.token_embedding.weight')
        state_dict.pop('text_model_embeddings.position_embedding.weight')

        torch.save(state_dict, file_name)