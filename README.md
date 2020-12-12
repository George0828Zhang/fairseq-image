## Setup

1. Install fairseq
```bash
git clone https://github.com/pytorch/fairseq.git
cd fairseq
git checkout 8a0b56efeecba6941c1c0e81a3b86f0a219e6653 # the commit for fairseq-0.10.1
export CUDA_HOME=/usr/local/cuda
python setup.py install
```
2. (Optional) Install apex
```bash
# note that apex requires that nvcc on system should be same version as that used to build torch 
# e.g. torch 1.7.0+cu110 ==> nvcc -V should be 11.0,
#      torch 1.6.0+cu102 ==> nvcc -V should be 10.2, etc.

git clone https://github.com/NVIDIA/apex
cd apex
pip install -v --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" \
  --global-option="--deprecated_fused_adam" --global-option="--xentropy" \
  --global-option="--fast_multihead_attn" ./
```
3. Install dependencies
```bash
pip install -r requirements.txt
```

## Usage
**Download the dataset for training:**

```bash
mkdir -p DATA/download
cd DATA/download
wget http://lstm.seas.harvard.edu/latex/data/im2latex_validate_filter.lst
wget http://lstm.seas.harvard.edu/latex/data/im2latex_train_filter.lst
wget http://lstm.seas.harvard.edu/latex/data/im2latex_test_filter.lst
wget http://lstm.seas.harvard.edu/latex/data/formula_images_processed.tar.gz
wget http://lstm.seas.harvard.edu/latex/data/im2latex_formulas.norm.lst
tar -zxvf formula_images_processed.tar.gz
```

**Preprocess:**
```bash
cd ../../
DLDIR=./DATA/download
OUTDIR=./DATA/data-bin/im2latex
# optionally set fairseq directory, where we can find ${fairseq_dir}/scripts/spm_train.py & spm_encode.py 
# export fairseq_dir=./fairseq 
bash datasets/prepare-im2latex.sh ${DLDIR} ${OUTDIR} 
```

**Train**
```bash
cd exp
# optionally tune hyper parameters in train_insertion.sh
# then begin training
bash train_insertion.sh
```