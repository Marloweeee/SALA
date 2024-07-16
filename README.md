
<p align="center"> <img src='Figs/latest_fig1.png' align="center" height="540px"> </p>



### 1.Prepare the environment

``` python
python==3.8.10
icecream==2.1.3
matplotlib==3.7.5
mmcv_full==1.7.2
mmdet==2.11.0
einops==0.8.0
icecream==2.1.2
numpy==1.22.3
opencv_python==4.9.0.80
scipy==1.8.0
ftfy==6.1.1
timm==1.0.7
torch==2.0.0
torchvision==0.15.1
tqdm==4.65.2
transformers==4.39.3
```

The above is a tested environment. Other version of these packages may also be fine.

We recommmand to install mmdet from the source codes inside this repository (```./models/swin_model```).

### 2.Dataset preparation
We follow the data preparation of TransVG, which can be found in [GETTING_STARTED.md](https://github.com/djiajunustc/TransVG/blob/main/docs/GETTING_STARTED.md).

The download links of ReferItGame are broken.  Thus we upload the data splits and images to [Google Drive](https://drive.google.com/drive/folders/1D4shieeoKly6FswpdjSpaOrxJQNKTyTv?usp=sharing).

If the above link is not reachable, you can also download the data through this [answer](https://github.com/LeapLabTHU/Pseudo-Q/issues/2#issuecomment-1148624317)
### 3.Checkpoint preparation
```
mkdir checkpoints
```
You can set the ```--bert_model``` to ```bert-base-uncased``` to download bert checkpoints online or put ```bert-base-uncased``` into ```checkpoints/``` manually.

To train our model on refcoco/refcoco+/refcocog datasets, you need checkpoints trained on MSCOCO that the overlapping images of test set are excluded. We provide pretrained checkpoints on [Google Drive](https://drive.google.com/drive/folders/1GTi32iEfsJdYNtcHCUQIbhMdL5YFByVF?usp=sharing). For referit/flickr datasets, you can simply use the pretrained checkpoint from [Swin-Transformer](https://github.com/SwinTransformer/Swin-Transformer-Object-Detection).

## Training and Evaluation

### 1. Training

We present bash scripts for training  on refcoco as follows

```bash
dataset="unc"

python -m torch.distributed.launch --nproc_per_node=8 --use_env train.py --data_root ./ln_data/ \
 --batch_size 32 --lr 0.0001 --num_workers=8 \
 --output_dir ./outputs/$dataset \
 --dataset $dataset --max_query_len 20 \
 --aug_crop --aug_scale --aug_translate \
 --lr_drop 60 --epochs 90 --swin_checkpoint ./checkpoints/QRNet/unc_latest.pth

```

For training
```
conda activate conda_env

./train_refcoco.sh
```

It's similar to train the model on the other datasets. Differents is that on RefCOCOg, we recommend to set ```--max_query_len 40```, on RefCOCO+ We recommend to set ```--lr_drop 120```.

### 2.Evaluation

We present bash scripts for testing  on refcoco as follows
```bash
dataset="unc"
splits="test"
max_query_len=20
weight_file="./outputs/$dataset/test_path"
output_dir="./outputs/$dataset_$split"

python -m torch.distributed.launch --nproc_per_node=8 --use_env eval.py \
                --batch_size 64 --data_root ./ln_data/ \
                --dataset $dataset --max_query_len $max_query_len --eval_set $split \
                --eval_model $weight_file --output_dir  "$output_dir"
```

For  evaluation
```
conda activate conda_env

./eval_refcoco.sh
```
