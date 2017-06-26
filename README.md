# Deep Reinforcement Learning Tutorial

Contains Jupyter notebooks associated with the [*Deep Reinforcement Learning Tutorial*](https://conferences.oreilly.com/artificial-intelligence/ai-ny/public/schedule/detail/59390) given at the O'Reilly 2017 NYC AI Conference.

Required Unity Environments can be downloaded [here](https://drive.google.com/drive/folders/0BxZSPcA0DrkfQ2pPWkRFQkNiTnc?usp=sharing). Download the zip associated with your OS (ie Linux, Mac, or Windows) and unzip the files to the root directory of this repo.

All notebooks and environments tested with Python2 and Python3 on macOS Sierra.

## Requirements
* Tensorflow
* Pillow
* Matplotlib
* numpy
* scipy
* Jupyter

To install dependencies, run:

`pip install -r requirements.txt`

or 

`pip3 install -r requirements.txt`

## Training RL Agents

To launch jupyter, run:

`jupyter notebook`

Then navigate to `localhost:8888` to access each training notebook.

To monitor training progress, run the following from the root directory of this repo:

`tensorboard --logdir='./summaries`

Then navigate to `localhost:6006` to monitor progress with Tensorboard.
