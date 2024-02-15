# Multi-Domain Image-to-Image Translation using Transformers


### Environment
* Python 3.8, PyTorch 1.11.0


### Datasets
* ADE20k
* INIT [[dataset]](https://zhiqiangshen.com/projects/INIT/index.html)
* OUTSIDE15k 

### Training & Test Script
#### Training
Adjust setting in ```config/config.yaml```
- Train with a single GPU
```python
torchrun --standalone --nnodes=1  --nproc_per_node=1 train.py
```
```python
torchrun --nproc_per_node='num_gpus' train.py
```

- test

```python
test.py
```




### Config Path

config/config.yaml


