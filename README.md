mnist: training in Lenux and inference in Windows
====

Overview

- Train mnist using Lenux + python + tensorflow
- Inference in Windows + C language. 

## Requirement

- Ubuntu 18.04 LTS
- python 3.6.9
- tensorflow 1.14.0
- Windows 10
- Visual Studio 2017 Community

## Usage

1. save mnist raw data to your local like:  
./data/raw/t10k-images.idx3-ubyte  
./data/raw/t10k-labels.idx1-ubyte  
./data/raw/train-images.idx3-ubyte  
./data/raw/train-labels.idx1-ubyte  
1. start get_image.py
1. start configurate_data.py
1. start train.py
1. start keras_to_tensorflow.py to convert h5 to pb
1. copy pb and ./data/testImage.txt to Windows

## Licence

[MIT](https://github.com/tcnksm/tool/blob/master/LICENCE)