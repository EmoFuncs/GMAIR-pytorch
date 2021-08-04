# GMAIR-pytorch
An official implementation of GMAIR.

## Prepare
This project uses Python 3.8 and Pytorch 1.8.1.
```
git clone https://github.com/EmoFuncs/GMAIR-pytorch.git
pip install -r requirements.txt
```

Build bbox:
```
cd GMAIR-pytorch/gmair/utils/bbox
python setup.py build
cp build/lib/bbox.so .
```

## Datasets
### MultiMNIST dataset
[link](https://drive.google.com/file/d/1BIzWAExc0NDSF_a6RnTvfBMvbXhTAns5/view?usp=sharing)
The dataset is generated from a modified version of [MultiDigitMNIST](https://github.com/yonkshi/MultiDigitMNIST).

### Fruit2D dataset
[train images](https://drive.google.com/file/d/1MCXo6VRI6Pf8WG2-dHbPNCJZVKOpNoHX/view?usp=sharing)
[train annotations](https://drive.google.com/file/d/1wbidjghjwLracHq8HRZ-zidWIE0R4xSV/view?usp=sharing)
[test images](https://drive.google.com/file/d/11BDgxjnZ7wXwCPFksL4rHIthuddhLWUW/view?usp=sharing)
[test annotations](https://drive.google.com/file/d/13Y5ZRu5ojspYOI0Ku1nJ0tu1lPbFrlZa/view?usp=sharing)

Note that annotations are only used for evaluation.



## Train
For MultiMNIST, download MultiMNIST dataset. Unzip it, and put it into 'data/multi_mnist/'.
Substitute 'config.py' with 'mnist_config.py' in 'gmair/config'
```
cd gmair/config
cp mnist_config.py config.py
cd ../..
```

For Fruit2d, download Fruit2d dataset. Unzip them, and put them into 'data/fruit2d/'.
Substitute 'config.py' with 'fruit_config.py' in 'gmair/config'
```
cd gmair/config
cp mnist_config.py config.py
cd ../..
```

The architecture should be:
```
data
|---fruit2d
|   |---test_images
|   |   |---x.png
|   |
|   |---test_labels
|   |   |---x.txt
|   |
|   |---train_images
|   |   |---y.png
|   |
|   |---train_labels
|       |---y.txt
|   
|---scatter_mnist
    |---scattered_mnist_128x128_obj14x14.hdf5
```

Then,
```
python train.py
```



## Test
Checkpoints:
[MultiMNIST](https://drive.google.com/file/d/11VHRFyAE0K3Gstj0hdd8yzy4BVzUErxX/view?usp=sharing)
[Fruit2D](https://drive.google.com/file/d/13wG_gNqgBollwH1WLP_MTyxodIX-1dPE/view?usp=sharing)

Set the path of checkpoint file in the configuration file 'config.py' (the variable 'test_model_path').

```
python test.py
```
