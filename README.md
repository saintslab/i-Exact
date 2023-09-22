This is the official codes for [Activation Compression of Graph Neural Networks using Block-wise Quantization and Improved Variance Minimization](https://arxiv.org/abs/2309.11856). Note that this repository is heavily inspired by the [EXACT repository](https://github.com/warai-0toko/Exact).

## Install
This code is tested with Python 3.8 and CUDA 11.1. The environment for the code is the same as the one used in EXACT. In order to construct this environment, run the following commands:

```bash
conda create -n i-exact python=3.8 cudatoolkit=11.1
conda activate i-exact
pip install https://download.pytorch.org/whl/cu111/torch-1.9.0%2Bcu111-cp38-cp38-linux_x86_64.whl
pip install torch_scatter==2.0.8 torch_sparse==0.6.12  -f https://data.pyg.org/whl
/torch-1.9.0+cu111.html
pip install torch_geometric==1.7.2
pip install PyYAML
pip install ogb==1.3.1
pip install carbontracker
cd exact
pip install -v -e .
```

## Reproduce results

### Reproduce ogbn-arxiv results.
```bash
cd mem_speed_bench
python ./arxiv/train_full_batch.py --conf ./arxiv/conf/$MODEL.yaml --n_bits $BIT_WIDTH --kept_frac $FRAC --col_size $GROUP_SIZE --lo $ALPHA
```
### Important note
Currently, only the sage model is tested for I-exact. Note also that non-uniform quantization (equivalent to ALPHA != 1.0) has only been tested with BIT_WIDTH == 2.

MODEL must be chosen from {gcn, sage, gcn2, gat}, BIT_WIDTH must be chosen from {1,2,4,8}, FRAC is pretty flexible. it can be any float-point number <= 1.0. If FRAC == 1.0, then the random projection will not be applied. GROUP_SIZE can be any natural number, and is denoted by G/R in the paper. ALPHA denotes the width of the first quantization bin. Since the quantization bins are assumed to be symmetric around the middle of the support, this also defines the remaining two quantization bins. 

If you do not want to apply any quantization, you can change the commend to 
```
python ./arxiv/train_full_batch.py --conf ./arxiv/conf/$MODEL.yaml --act_fp --kept_frac $FRAC --col_size $GROUP_SIZE --lo $ALPHA
```

### Reproduce Flickr results.
For full-batch training, 
```bash
cd mem_speed_bench
python ./non_ogbn_datasets/train_full_batch.py --conf ./non_ogbn_datasets/conf/$MODEL.yaml --n_bits $BIT_WIDTH --kept_frac $FRAC --dataset flickr --grad_norm 0.5 --col_size $GROUP_SIZE --lo $ALPHA
```
MODEL must be chosen from {gcn, sage, gcn2}. BIT_WIDTH must be chosen from {1,2,4,8}, FRAC can be any float-point number <= 1.0. 

### Get the occupied memory and training throughout.
Add the flag **--debug_mem** and **--test_speed** to the above commends. For example,
```
python ./arxiv/train_full_batch.py --conf ./arxiv/conf/$MODEL.yaml --n_bits $BIT_WIDTH --kept_frac $FRAC --debug_mem --test_speed
```
