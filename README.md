# SD4Match: Learning to Prompt Stable Diffusion Model for Semantic Matching

**[Project Page](http://sd4match.active.vision/) | [Arxiv](https://arxiv.org/abs/2310.17569)**

[Xinghui Li<sup>1</sup>](https://scholar.google.com/citations?user=XLlgbBoAAAAJ&hl=en),
Jingyi Lu<sup>2</sup>, 
[Kai Han<sup>2</sup>](https://www.kaihan.org/), 
[Victor Prisacariu<sup>1</sup>](https://www.robots.ox.ac.uk/~victor//)

[<sup>1</sup>Active Vision Lab, University of Oxford](https://www.robots.ox.ac.uk/~lav/)&nbsp;&nbsp;&nbsp;
[<sup>2</sup>Visual AI Lab, University of Hong Kong](https://visailab.github.io/)

## Environment
The environment can be easily installed through [conda](https://docs.conda.io/projects/miniconda/en/latest/) and pip. After downloading the code, run the following command:
```shell
$conda create -n sd4match python=3.10
$conda activate sd4match

$conda install pytorch==1.13.1 torchvision==0.14.1 torchaudio==0.13.1 pytorch-cuda=11.6 -c pytorch -c nvidia
$conda install xformers -c xformers
$pip install yacs pandas scipy einops matplotlib triton timm diffusers accelerate transformers datasets tensorboard pykeops scikit-learn
```

## Data
#### PF-Pascal
1. Download PF-Pascal dataset from [link](https://www.di.ens.fr/willow/research/proposalflow/).
2. Rename the outermost directory from `PF-dataset-PASCAL` to `pf-pascal`.
3. Download lists for image pairs from [link](https://www.robots.ox.ac.uk/~xinghui/sd4match/pf-pascal_image_pairs.zip).
4. Place the lists for image pairs under `pf-pascal` directory. The structure should be:
```
pf-pascal
├── __MACOSX
├── PF-dataset-PASCAL
├── trn_pairs.csv
├── val_pairs.csv
└── test_pairs.csv
```
#### PF-Willow
1. Download PF-Willow dataset from the [link](https://www.di.ens.fr/willow/research/proposalflow/).
2. Rename the outermost directory from `PF-dataset` to `pf-willow`.
3. Download lists for image pairs from [link](https://www.robots.ox.ac.uk/~xinghui/sd4match/test_pairs.csv).
4. Place the lists for image pairs under `pf-willow` directory. The structure should be:
```
pf-willow
├── __MACOSX
├── PF-dataset
└── test_pairs.csv
```
#### SPair-71K
1. Download SPair-71K dataset from [link](https://cvlab.postech.ac.kr/research/SPair-71k/). After extraction,  No more action required.

## Setup
1. Create symbol links to PF-Pascal, PF-Willow and SPair-71k dataset in `asset` directory. This can be done by:
```
ln -s /your/path/to/pf-pascal asset/pf-pascal
ln -s /your/path/to/pf-willow asset/pf-willow
ln -s /your/path/to/SPair-71k asset/SPair-71k
```
2. Create a directory named `sd4match` under `asset`. This is to save pre-computed features, checkpoints and learned prompts.
```
# create sd4match directly
mkdir asset/sd4match

# or create sd4match at anywhere you want and use symbol link
ln -s /your/path/to/sd4match asset/sd4match
```

3. Run `pre_compute_dino_feature.py`. This would pre-compute DINOv2 feature for all images in PF-Pascal, PF-Willow and SPair-71k and save them in `asset/sd4match`. The structure should be:
```
sd4match
└── asset
    └── DINOv2
        ├── pfpascal
        |   └── cached_output.pt
        ├── pfwillow
        |   └── cached_output.pt
        └── spair
            └── cached_output.pt
```

## Training
The bash scripts for training are provided in `script` directory, and organized based on training data and prompt type.

For example, to train `SD4Match-CPM` on SPair-71k dataset, run:
```
cd script/spair
sh sd4match_cpm.sh
```
The batch size per GPU is currently set to `3`, which would take about `22G` GPU memory to train. Reduce the batch size if necessary. The training script will generate two directories in `asset/sd4match`: `log` and `prompt`. Tensorboard logs and training states are saved in `log`, and learned prompts are saved in `prompt`. For example, training `SD4Match-CPM` on SPair-71k dataset will generate:
```
sd4match
├── asset
|   ├── ...
├── log
|   └── spair
|       └── CPM_spair_sd2-1_Pair-DINO-Feat-G25-C50_constant_lr0.01
|           └── ...(Tensorboard log and training states)
└── prompt
    └── CPM_spair_sd2-1_Pair-DINO-Feat-G25-C50
        └── ckpt.pt
```

## Testing
To replicate our results reported in the paper on SPair-71k, either learning the prompt by yourself or downloading our pre-trained prompt and place them under `asset/sd4match/prompt` directory. Run:
```
python test.py \ 
--dataset spair \
--prompt_type $PROMPT_NAME \
--timestep 50 \
--layer 1
```
Replace `$PROMPT_NAME` with prompt your want. It needs to have a corresponding directory under `asset/sd4match/prompt`. For example, to evaluate `SD4Match-CPM`, run:
```
python test.py \ 
--dataset spair \
--prompt_type CPM_spair_sd2-1_Pair-DINO-Feat-G25-C50 \
--timestep 50 \
--layer 1
```

## Pretrained Prompt
Our pretrained prompt can be downloaded through [link](https://www.robots.ox.ac.uk/~xinghui/sd4match/pretrained_prompts.zip).

## Acknowledgement
[Kai Han](https://www.kaihan.org/) is supported by Hong Kong Research
Grant Council - Early Career Scheme (Grant No. 27208022), National Natural Science Foundation of
China (Grant No. 62306251), and HKU Seed Fund for Basic Research.

We also sincerely thank [Zirui Wang](https://scholar.google.com/citations?user=zCBKqa8AAAAJ&hl=en) for his inspiring discussion.

## Citation
```
@misc{li2023sd4match,
	title={SD4Match: Learning to Prompt Stable Diffusion Model for Semantic Matching}, 
	author={Xinghui Li and Jingyi Lu and Kai Han and Victor Prisacariu},
	year={2023},
	eprint={2310.17569},
	archivePrefix={arXiv},
	primaryClass={cs.CV}
    }
```