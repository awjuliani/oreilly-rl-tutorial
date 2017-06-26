# Deep Reinforcement Learning Tutorial

Contains Jupyter notebooks associated with the [*Deep Reinforcement Learning Tutorial*](https://conferences.oreilly.com/artificial-intelligence/ai-ny/public/schedule/detail/59390) given at the O'Reilly 2017 NYC AI Conference.

Required Unity Environments can be downloaded [here](https://drive.google.com/drive/folders/0BxZSPcA0DrkfQ2pPWkRFQkNiTnc?usp=sharing).

All notebooks and environments tested with Python2 on macOS Sierra.

## Requirements
* Tensorflow
* Pillow
* Matplotlib
* numpy
* scipy
* Jupyter

To install dependencies, run:

`pip install -r requirements.txt`

To launch jupyter, run:

`jupyter notebook`

Then navigate to `localhost:8888` to access each training notebook.

## Training Models

To monitor training progress, run the following from the root directory of this repo:

`tensorboard --logdir='./`

Then navigate to `localhost:6006` to monitor progress with Tensorboard.
