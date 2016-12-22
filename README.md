# tensorflow-vgg
Re-implementation of VGG Network in tensorflow

# setup

```
pip install pyyaml
pip install skimage
pip install skdata
pip install tensorflow
```

# training

```
python train.py experiment.yaml
```

# training on multiple gpus

```
python train_multi_gpu.py experiment.yaml
```

# prediction

```
python predict.py dog.jpg
```
