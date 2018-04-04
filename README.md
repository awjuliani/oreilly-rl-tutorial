# Deep Reinforcement Learning Tutorial

Contains Jupyter notebooks associated with the [*Deep Reinforcement Learning Tutorial*](https://ai.oreilly.com.cn/ai-cn/public/schedule/detail/64627?locale=en) given at the O'Reilly 2018 Beijing AI Conference. Slides from the presentation can be downloaded [here](https://drive.google.com/open?id=0BxZSPcA0DrkfNG9aSjYxM1RMVzQ).

Required Unity Environments can be downloaded [here](https://drive.google.com/drive/folders/0BxZSPcA0DrkfQ2pPWkRFQkNiTnc?usp=sharing). Download and unzip the .zip file associated with your OS (ie Linux, Mac, or Windows) and move each of the files within the unzipped folder (ie 2DBall, 3DBall, etc) to the root directory of this repository.

## Requirements
* Python 3
* Tensorflow (version 1.0+)
* Pillow
* Matplotlib
* numpy
* scipy
* Jupyter

To install dependencies, run:

`pip3 install -r requirements.txt`

If your Python environment doesn't include `pip3`, see these [instructions](https://packaging.python.org/guides/installing-using-linux-tools/#installing-pip-setuptools-wheel-with-linux-package-managers) on installing it.

## Training RL Agents

To launch jupyter, run:

`jupyter lab` 

Then navigate to `localhost:8888` to access each training notebook.

To monitor training progress, run the following from the root directory of this repo:

`tensorboard --logdir='./summaries'`

Then navigate to `localhost:6006` to monitor progress with Tensorboard.

## Troubleshooting

### macOS Permission Error

If you recieve a permission error when attempting to launch an environment on macOS, run:

`chmod -R 755 *.app` 
