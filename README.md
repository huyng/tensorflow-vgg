# tensorflow-vgg
Re-implementation of VGG Network in tensorflow

### setup

```
pip install pyyaml
pip install skimage
pip install skdata
pip install tensorflow-gpu
```

### training

```
python train_simple.py experiment.yaml
```

### training on multiple gpus

```
python train_parallel.py experiment.yaml
```

### prediction

```
python predict.py --weights=exp0/checkpoints/weights.500.npz dog.jpg
```
