### Installation
1. Install fairseq
```
git clone https://github.com/pytorch/fairseq.git
cd fairseq
git checkout 8a0b56efeecba6941c1c0e81a3b86f0a219e6653
export CUDA_HOME=/usr/local/cuda
pip install .
```
2. (Optional) Install apex
```
git clone https://github.com/NVIDIA/apex
cd apex
pip install -v --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" \
  --global-option="--deprecated_fused_adam" --global-option="--xentropy" \
  --global-option="--fast_multihead_attn" ./
```
3. Install dependencies
```
pip install -r requirements.txt
```