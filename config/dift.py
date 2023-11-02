from yacs.config import CfgNode as CN

_CN = CN()
 
# dataset configuration
_CN.DATASET = CN()
_CN.DATASET.NAME = 'spair'
_CN.DATASET.ROOT = 'asset/'          # '/home/xinghui/storage'
_CN.DATASET.IMG_SIZE = 768
_CN.DATASET.MEAN = [0.5, 0.5, 0.5]
_CN.DATASET.STD = [0.5, 0.5, 0.5]

# stable diffusion configuration
_CN.STABLE_DIFFUSION = CN()
_CN.STABLE_DIFFUSION.VERSION = '2-1'
_CN.STABLE_DIFFUSION.SAVE_MEMORY = True             # if True, less memory used, but lower speed

# feature extractor configuration
_CN.FEATURE_EXTRACTOR = CN()
_CN.FEATURE_EXTRACTOR.METHOD = 'dift'                # select between ('dift', 'sd-dino')
_CN.FEATURE_EXTRACTOR.SELECT_TIMESTEP = 261          # select from 1-1000
_CN.FEATURE_EXTRACTOR.SELECT_LAYER = 1               # if use 'dift': select from (0,1,2,3). if use 'sd-dino': select from (0,1,...,11)
_CN.FEATURE_EXTRACTOR.ENSEMBLE_SIZE = 8              # if ensemble_size > 1, the denosing processed are repeated and the feature is the average over multiple trials

_CN.FEATURE_EXTRACTOR.FUSE_DINO = False
_CN.FEATURE_EXTRACTOR.ENABLE_L2_NORM = False

_CN.FEATURE_EXTRACTOR.PROMPT_TYPE = 'text'
cfg = _CN