from yacs.config import CfgNode as CN

_CN = CN()

# dataset configuration
_CN.DATASET = CN()
_CN.DATASET.NAME = 'spair'
_CN.DATASET.ROOT = 'asset/'
_CN.DATASET.IMG_SIZE = 768
_CN.DATASET.MEAN = [0.5, 0.5, 0.5]
_CN.DATASET.STD = [0.5, 0.5, 0.5]

# stable diffusion configuration
_CN.STABLE_DIFFUSION = CN()
_CN.STABLE_DIFFUSION.VERSION = '1-5'
_CN.STABLE_DIFFUSION.SAVE_MEMORY = True              # if True, less memory used, but lower speed

# feature extractor configuration
_CN.FEATURE_EXTRACTOR = CN()
_CN.FEATURE_EXTRACTOR.METHOD = 'dift'                # select between ('dift', 'sd-dino')
_CN.FEATURE_EXTRACTOR.SELECT_TIMESTEP = 261          # select from 1-1000
_CN.FEATURE_EXTRACTOR.SELECT_LAYER = 1               # if use 'dift': select from (0,1,2,3). if use 'sd-dino': select from (0,1,...,11)
_CN.FEATURE_EXTRACTOR.ENSEMBLE_SIZE = 2              # if ensemble_size > 1, the denosing processed are repeated and the feature is the average over multiple trials

_CN.FEATURE_EXTRACTOR.PROMPT_TYPE = 'text'
_CN.FEATURE_EXTRACTOR.ASSET_ROOT = "asset/sd4match/asset"      # root to cached asset, like cached clip or dino feature
_CN.FEATURE_EXTRACTOR.PROMPT_CACHE_ROOT = "asset/sd4match/prompt"
_CN.FEATURE_EXTRACTOR.LOG_ROOT = "asset/sd4match/log"

_CN.FEATURE_EXTRACTOR.ENABLE_L2_NORM = True
_CN.FEATURE_EXTRACTOR.FUSE_DINO = False

# Evaluator configuration
_CN.EVALUATOR = CN()
_CN.EVALUATOR.ALPHA = [0.05, 0.1, 0.15]
_CN.EVALUATOR.BY = 'image'                          # select between ('image', 'point'), PCK per image or PCK per point

def get_default_defaults():
    """Get a yacs CfgNode object with default values for my_project."""
    # Return a clone so that the defaults will not be altered
    # This is for the "local variable" use pattern
    return _CN.clone()

def convert_config_to_dict(config):
    if not isinstance(config, CN):
        return config
    return {k: convert_config_to_dict(v) for k, v in config.items()}