import torch
import numpy as np

from .geometry import *
from .matching import *


class PCKEvaluator:

    def __init__(self, cfg) -> None:
        
        self.alpha = cfg.EVALUATOR.ALPHA
        self.by = cfg.EVALUATOR.BY
        self.method_options = ('nn', 'bilinear', 'softmax', 'kernelsoftmax')

        self.result = {}
        for method_name in self.method_options:
            for alpha in self.alpha:
                self.result.update({f'{method_name}_pck{alpha}': {"all": []}})

    def clear_result(self):
        self.result = {key : {'all': []} for key in self.result}


    def calculate_pck(self, trg_kps, matches, n_pts, categories, pckthres, method_name):
        '''
        trg_kps [torch.Tensor] (BxNx2)
        matches [torch.Tensor] (BxNx2)
        n_pts[torch.Tensor] (B)
        pckthres[float] (B)
        '''
        B = trg_kps.shape[0]
        
        for b in range(B):
            npt = n_pts[b].item()
            thres = pckthres[b].item()
            category = categories[b]

            tkps = trg_kps[b, :npt]     # npt x 2
            mats = matches[b, :npt]     # npt x 2

            diff = torch.norm(tkps-mats, dim=-1)
            
            for alpha in self.alpha:
                if self.by == 'image':
                    pck = (diff <= alpha*thres).float().mean()
                    if category in self.result[f'{method_name}_pck{alpha}']:
                        self.result[f'{method_name}_pck{alpha}'][category].append(pck.item())
                    else:
                        self.result[f'{method_name}_pck{alpha}'][category] = []
                        self.result[f'{method_name}_pck{alpha}'][category].append(pck.item())
                    self.result[f'{method_name}_pck{alpha}']["all"].append(pck.item())
                elif self.by == "point":
                    pck = (diff <= alpha*thres).float().tolist()
                    if category in self.result[f'{method_name}_pck{alpha}']:
                        self.result[f'{method_name}_pck{alpha}'][category].extend(pck)
                    else:
                        self.result[f'{method_name}_pck{alpha}'][category] = []
                        self.result[f'{method_name}_pck{alpha}'][category].extend(pck)
                    self.result[f'{method_name}_pck{alpha}']["all"].extend(pck)
                else:
                    raise ValueError(f"select between ('image', 'point')")

    def evaluate_feature_map(self, batch, softmax_temp=0.04, gaussian_suppression_sigma=7, enable_l2_norm=True):
        '''
        Given source and target feature map, paired with src_kps and trg_kps, this function evaluates feature map's 
        top_k accuracy on finding correspondence of src_kps within top k nearest neighbours. 
        IN:
            batch['src_img'][torch.Tensor] (B x 3 x H1 x W2)
            batch['trg_img'][torch.Tensor] (B x 3 x H2 x W2)
            batch['src_featmaps'][torch.Tensor] (B x C x h1 x w1)
            batch['trg_featmaps'][torch.Tensor] (B x C x h2 x w2)
            batch['src_kps'][torch.Tensor] (B x Nq x 2)
            batch['trg_kps'][torch.Tensor] (B x Nq x 2)
            batch['n_pts'][torch.Tensor] (B)
            batch['pckthres][torch.Tensor] (B)
            alpha[float]
            softmax_temp[float]: softmax temperature for matching_strategy being softargmax. For automatic temperature, it has no effect
        '''
        src_img = batch['src_img'].clone()
        trg_img = batch['trg_img'].clone()
        src_featmaps = batch['src_featmaps'].clone()
        trg_featmaps = batch['trg_featmaps'].clone()
        src_kps = batch['src_kps'].clone()
        trg_kps = batch['trg_kps'].clone()
        n_pts = batch['n_pts'].clone()
        categories = batch['category']
        pckthres = batch['pckthres'].clone()

        # get image and featmap size
        H1, W1 = src_img.shape[2:]
        H2, W2 = trg_img.shape[2:]
        h1, w1 = src_featmaps.shape[2:]
        h2, w2 = trg_featmaps.shape[2:]

        # find matches depending on matching_strategy
        src_kps = scaling_coordinates(src_kps, (H1, W1), (h1, w1))

        bilinear_matches = bilinear_get_matches(src_featmaps, trg_featmaps, src_kps, l2_norm=enable_l2_norm)
        nn_matches = nn_get_matches(src_featmaps, trg_featmaps, src_kps, l2_norm=enable_l2_norm)
        softargmax_matches = softargmax_get_matches(src_featmaps, trg_featmaps, src_kps, softmax_temp, l2_norm=enable_l2_norm)
        kernelsoftargmax_matches = kernel_softargmax_get_matches(src_featmaps, trg_featmaps , src_kps, softmax_temp, gaussian_suppression_sigma, l2_norm=enable_l2_norm)

        bilinear_matches = scaling_coordinates(bilinear_matches, (h2, w2), (H2, W2))
        nn_matches = scaling_coordinates(nn_matches, (h2, w2), (H2, W2))
        softargmax_matches = scaling_coordinates(softargmax_matches, (h2, w2), (H2, W2))
        kernelsoftargmax_matches = scaling_coordinates(kernelsoftargmax_matches, (h2, w2), (H2, W2))

        self.calculate_pck(trg_kps, bilinear_matches, n_pts, categories, pckthres, 'bilinear')
        self.calculate_pck(trg_kps, nn_matches, n_pts, categories, pckthres, 'nn')
        self.calculate_pck(trg_kps, softargmax_matches, n_pts, categories, pckthres, 'softmax')
        self.calculate_pck(trg_kps, kernelsoftargmax_matches, n_pts, categories, pckthres, 'kernelsoftmax')

    def summerize_result(self):
        out = {}
        for method_name in self.method_options:
            for alpha in self.alpha:
                out[f'{method_name}_pck{alpha}'] = {}
                for k, v in self.result[f'{method_name}_pck{alpha}'].items():
                    out[f'{method_name}_pck{alpha}'][k] = np.array(v).mean()
        return out
    
    def print_summarize_result(self):
        result = self.summerize_result()
        print(" " * 16 + "".join([f"{alpha:<10}" for alpha in self.alpha]))  # header
        for method_name in self.method_options:
            pcks = [f"{result[f'{method_name}_pck{alpha}']['all']:.2%}" for alpha in self.alpha]
            row = f"{method_name:<15}" + "".join([f"{pck:<10}" for pck in pcks])
            print(row)  # rows

    def save_result(self, save_file):
        result = self.summerize_result()
        outstring = ""
        for method_name in self.method_options:
            outstring += f"{method_name}:\n"
            catstring = ""
            for alpha in self.alpha:
                cat_list = []
                pck_list = []
                for k, v in result[f'{method_name}_pck{alpha}'].items():
                    if k != "all":
                        cat_list.append(k)
                        pck_list.append(v)
                cat_list = np.array(cat_list)
                pck_list = np.array(pck_list)
                indices = np.argsort(cat_list)
                cat_list = cat_list[indices]
                pck_list = pck_list[indices]
                cat_list = cat_list.tolist()
                pck_list = pck_list.tolist()
                pck_list = [f"{pck:.2%}" for pck in pck_list]
                cat_list.append("all")
                pck_list.append(f"{result[f'{method_name}_pck{alpha}']['all']:.2%}")

                if len(catstring) == 0:
                    catstring += " " * 12 + "".join([f"{category:<12}" for category in cat_list]) + "\n"
                    outstring += catstring
                row = f"{alpha:<12}" + "".join([f"{pck:<12}" for pck in pck_list]) + "\n"
                outstring += row

            outstring += "-----------------------------------------------------------------\n"

        with open(save_file, "w") as f:
            f.write(outstring)
