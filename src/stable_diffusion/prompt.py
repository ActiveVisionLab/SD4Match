import os
import torch
from typing import List, Dict


class PromptManager:
    """
    For visual-captioning prompt method, We pre-compute captions for all images to save memory and speed.
    """

    def __init__(self, cfg) -> None:

        self.cfg = cfg

        prompt_type = cfg.FEATURE_EXTRACTOR.PROMPT_TYPE
        prompt_cache_root = cfg.FEATURE_EXTRACTOR.PROMPT_CACHE_ROOT

        if "HiddenState" in prompt_type or "TokenEmbeds" in prompt_type:

            prompt_file = os.path.join(prompt_cache_root, prompt_type, "caption.pt")
            
            if os.path.exists(prompt_file):
                self.saved_prompt = torch.load(prompt_file)
            else:
                self.saved_prompt = None
            

    def __call__(self, 
                imnames: List[str], 
                class_names: List[str]):
        """
        IN:
            imnames [List[str]]: a list of string contains image names.
            class_names: a list of string contains class names of the image
        """
        prompt_type = self.cfg.FEATURE_EXTRACTOR.PROMPT_TYPE
        
        if prompt_type == "text":
            """Fixed template prompt"""
            prompt = []
            # Handle pfwillow cls name
            for cls in class_names:
                if "(" in cls:
                    if "car" in cls:
                        cls = "car"
                    elif "duck" in cls:
                        cls = 'duck'
                    elif "motorbike" in cls:
                        cls = "motorbike"
                    elif "winebottle" in cls:
                        cls = "winebottle"
                prompt.append(f"a photo of a {cls}")

        elif prompt_type == "empty":
            prompt = ["" for cls in class_names]
        
        elif "HiddenState" in prompt_type or "TokenEmbeds" in prompt_type:
            
            if self.saved_prompt is None:
                raise ValueError("saved_prompt is None!")

            prompt = []
            for cls, imname in zip(class_names, imnames):

                if "Class" in prompt_type:
                    prompt.append(self.saved_prompt[f"{cls}"])
                elif "Single" in prompt_type:
                    prompt.append(self.saved_prompt["dataset"])

            prompt = torch.cat(prompt, dim=0)

        else:
            raise NotImplementedError(f"Invalid prompt type: {self.cfg.FEATURE_EXTRACTOR.PROMPT_TYPE}.")
        
        return prompt