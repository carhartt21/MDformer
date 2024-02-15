# Multi-Domain Image-to-Image Translation using Transformers
![architecture](https://github.com/carhartt21/MDformer/assets/24622304/fb90b785-69d3-43f9-a9be-54760b44fc20)


### Environment
* Python 3.8, PyTorch 1.11.0


### Datasets
* [ADE20k](https://groups.csail.mit.edu/vision/datasets/ADE20K/)
* [Mapillary Vistas](https://research.mapillary.com/)
* [SkyCloud](https://ieeexplore.ieee.org/document/10270450)

### Training & Test Script
#### Training
Adjust setting in ```config/config.yaml```
- Train with a single GPU
```python
torchrun --standalone --nnodes=1  --nproc_per_node=1 train.py
```
- Distributed training using multiple GPUs
```python
torchrun --nproc_per_node='num_gpus' train.py
```

- Testing

```python
python test.py
```




### Config Path

config/config.yaml


