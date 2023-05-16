[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![PRs Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen.svg)](https://github.com/tirthajyoti/Papers-Literature-ML-DL-RL-AI/pulls)
<!-- [![GitHub forks](https://img.shields.io/github/forks/tirthajyoti/Deep-Learning-with-Python.svg)](https://github.com/tirthajyoti/Deep-Learning-with-Python/network)
[![GitHub stars](https://img.shields.io/github/stars/tirthajyoti/Deep-Learning-with-Python.svg)](https://github.com/tirthajyoti/Deep-Learning-with-Python/stargazers)
[![PRs Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen.svg)](https://github.com/tirthajyoti/Deep-Learning-with-Python/pulls) -->

# Collection and Implementation for ML/DL/RL/AI Algorithms
Collection and implementation of a variety of machine learning/deep learning/reinforcement learning code examples (notebooks and scripts), projects, and paper lists on various topics. 

Some of the Jupyter notebooks are referenced from Andrew Ng's **[Deep Learning Specialization](https://www.coursera.org/specializations/deep-learning)** , D2L_ai's **[D2L](https://d2l.ai/index.html)** and the Probabilistic Machine Learning book by Kevin Murphy.

## Requirements
If you wish to run my codes directly, your code environment may need to satisfy the following requirements:
* Python 3.8+
* NumPy (`pip install numpy` or `conda install numpy`)
* Pandas (`pip install pandas` or `conda install numpy`)
* Pytorch (`pip install torch` or `conda install numpy`)

A conda environment is recommended:
```bash
export PROJECT_DIR=<ABSOLUTE path to the repository root>
git clone https://github.com/clarenceluo78/deep-learning-lookup $PROJECT_DIR
cd $PROJECT_DIR

conda create -n dl-lookup python=3.8
conda activate dl-lookup

conda install pytorch==1.7.1 torchvision==0.8.2 cudatoolkit=10.1.243 numpy=1.19.2 -c pytorch -y
pip install -r requirements.txt

# if the following commands do not succeed, update conda
conda env config vars set PYTHONPATH=${PYTHONPATH}:${PROJECT_DIR}
conda env config vars set PROJECT_DIR=${PROJECT_DIR}

conda deactivate
conda activate dl-lookup
```


**NOTE**: Most of the Jupyter notebooks in this repo are built on **[Google Colaboratory](https://colab.research.google.com/)** using **[Google GPU cluster](https://cloud.google.com/gpu/)** and a virtual machine. Therefore, you may not need to install these packages on your local machine if you also want to use Google colab. You can **directly launch the notebooks in your Google colab environment by clicking on the links provided in the notebooks** (of course, that makes a copy of my notebook on to your Google drive).

> For more information about using **Google Colab** for your deep learning work, [check their FAQ here](https://research.google.com/colaboratory/faq.html). You are also welcomed to take a look at Kaggle notebook!



## Utility modules

### Utility module for example notebooks
A utility function file called `util.py` is created under `lib` directory. I use functions from this module whenever possible in the Jupyter notebooks. For more information you can look into the `lib` folder.

You can download another reference module (raw Python file) from here: [DL-Utility-Module](https://raw.githubusercontent.com/tirthajyoti/Deep-learning-with-Python/master/Notebooks/utils/DL_utils.py)

<!-- 
## Notebooks

### Deep learning vs. linear model
* We show a nonlinear function approximation task performed by linear model (polynomial degree) and a simple 1/2 hidden layer (densely connected) neural net to illustrate the difference and the capacity of deep neural nets to take advantage of larger datasets ([Here is the Notebook](https://github.com/tirthajyoti/Deep-learning-with-Python/blob/master/Notebooks/Function%20approximation%20by%20linear%20model%20and%20deep%20network.ipynb)).

### Demo of a general-purpose regression module
* We implemented a general-purpose trainer module for regression task with tabular datasets. The idea is that you can simply read a dataset (e.g. a CSV file), choose the input and target variables, build a densely-connected neural net, train, predict, and save the model for deployment. This the demo notebook for that module ([Here is the Notebook](https://github.com/tirthajyoti/Deep-learning-with-Python/blob/master/Notebooks/Demo_general_purpose_regression_module.ipynb)).

### Simple Conv Net
* [Fashion MNIST](https://github.com/zalandoresearch/fashion-mnist) image classification using densely connected network and 1/2/3 layer CNNs ([Here is the Notebook](https://github.com/tirthajyoti/Computer_vision/blob/master/Notebooks/Fashion_MNIST_using_CNN.ipynb)).

### Using Keras `ImageDataGenerator` and other utilities

* _Horse or human_ image classification using Keras `ImageDataGenerator` and **Google colaboratory** platform. ([Here is the Notebook](https://github.com/tirthajyoti/Computer_vision/blob/master/Notebooks/Horse_or_Human_with_ImageGenerator.ipynb))

* Classification on the [flowers dataset](https://www.kaggle.com/alxmamaev/flowers-recognition) and the famous [Caltech-101 dataset](http://www.vision.caltech.edu/Image_Datasets/Caltech101/) using `fit_generator` and `flow_from_directory()` method of the `ImageDataGenerator`. Illustrates how to streamline CNN model building from a single storage of image data using these utility methods. ([Here is the Notebook](https://github.com/tirthajyoti/Deep-learning-with-Python/blob/master/Notebooks/Keras_flow_from_directory.ipynb))

###  Transfer learning
* Simple illustration of [transfer learning](https://machinelearningmastery.com/transfer-learning-for-deep-learning/) using CIFAR-10 dataset ([Here is the Notebook](https://github.com/tirthajyoti/Deep-learning-with-Python/blob/master/Notebooks/Transfer_learning_CIFAR.ipynb))

* Transfer learning with the famous [Inception v3 model](https://www.analyticsvidhya.com/blog/2018/10/understanding-inception-network-from-scratch/) - building a classifier of pneumonia from chest X-ray images. ([Here is the Notebook](https://github.com/tirthajyoti/Deep-learning-with-Python/blob/master/Notebooks/Transfer-learning-InceptionV3.ipynb))

### Activation maps
* We illustrate how to show the activation maps of various layers in a deep CNN model with just a couple of lines of code using `Keract` library. ([Here is the Notebook](https://github.com/tirthajyoti/Deep-learning-with-Python/blob/master/Notebooks/Keract-activation.ipynb)) 

### Adding object-oriented programming style to deep learning workflow
* Adding simple [Object-oriented Programming (OOP)](https://realpython.com/python3-object-oriented-programming/) principle to your deep learning workflow ([Here is the Notebook](https://github.com/tirthajyoti/Computer_vision/blob/master/Notebooks/OOP_principle_deep_learning.ipynb)).

### Keras `Callbacks` using ResNet
* [ResNet](https://medium.com/@14prakash/understanding-and-implementing-architectures-of-resnet-and-resnext-for-state-of-the-art-image-cf51669e1624) on [CIFAR-10 dataset](https://www.cs.toronto.edu/~kriz/cifar.html), showing how to use Keras Callbacks classes like `ModelCheckpoint`, `LearningRateScheduler`, and `ReduceLROnPlateau`. You can also change a single parameter to generate ResNet of various depths. ([Here is the Notebook](https://github.com/tirthajyoti/Deep-learning-with-Python/blob/master/Notebooks/ResNet-on-CIFAR10.ipynb)).

### Simple RNN
* Time series prediction using simple RNN (a single RNN layer followed by a densely connected layer). We show that a complicated time-series signal is correctly predicted by a simple RNN even when trained with only 25% of the data. ([Here is the Notebook](https://github.com/tirthajyoti/Deep-learning-with-Python/blob/master/Notebooks/SimpleRNN-time-series.ipynb)) 

### Text generation using LSTM
* Automatic text generation (based on simple character vectors) using [LSTM network](https://colah.github.io/posts/2015-08-Understanding-LSTMs/). Play with character sequence length, LSTM architecture, and hyperparameters to generate synthetic texts based on a particular author's style! ([Here is the Notebook](https://github.com/tirthajyoti/Deep-learning-with-Python/blob/master/Notebooks/LSTM_text_gen_Dickens.ipynb)).

### Bi-directional LSTM for sentiment classification
* [Bi-directional LSTM with embedding](https://machinelearningmastery.com/develop-bidirectional-lstm-sequence-classification-python-keras/) applied to the IMDB sentiment classification task ([Here is the Notebook](https://github.com/tirthajyoti/Deep-learning-with-Python/blob/master/Notebooks/LSTM_bidirectional_IMDB_data.ipynb))

### Generative adversarial network (GAN)
* Simple demo of building a GAN model from scratch using a one-dimensional algebraic function ([Here is the Notebook](https://github.com/tirthajyoti/Deep-learning-with-Python/blob/master/Notebooks/GAN_1D.ipynb))

### Scikit-learn wrapper for Keras
* [Keras Scikit-learn wrapper](https://keras.io/scikit-learn-api/) example with 10-fold cross-validation and exhaustive grid search ([Here is the Notebook](https://github.com/tirthajyoti/Deep-learning-with-Python/blob/master/Notebooks/Keras_Scikit_Learn_wrapper.ipynb)) -->
