### Setup
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
# e.g. torch 1.6.0 ==> nvcc -V should be 10.2
# e.g. torch 1.2.0 ==> nvcc -V should be 10.0

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

### Usage
#### Train
```bash
cd exp
bash train_insertion.sh
```