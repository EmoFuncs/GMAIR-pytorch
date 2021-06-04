# GMAIR-pytorch
An official implementation of GMAIR [paper](https://arxiv.org/abs/2106.01722).

## Prepare
```
git clone https://github.com/EmoFuncs/GMAIR-pytorch.git
cd ./GMAIR-pytorch/gmair/utils/bbox
python setup.py build
```

## Datasets
The links to datasets will be released soon.
<!---
Download MultiMNIST from \[TODO\]
Download Fruit2D from \[TODO\]
-->

## Train
For training MultiMNIST or Fruit2D, substitute 'config.py' with 'mnist_config.py' or 'fruit_config.py' in folder './config', respectively.

```
python train.py
```

## Test
Set the path of checkpoint file in the configuration file 'config.py'.

```
python test.py
```
