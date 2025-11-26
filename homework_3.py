# %%
%load_ext autoreload
%autoreload 2

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# %% [markdown]
# **Group members:** [your names]

# %% [markdown]
# ## Part 1: Multi-Layer Perceptrons

# %% [markdown]
# ### (a)
# > Using the `mnist_iter()` method from `dl.py`, load and display some
# > datapoints from MNIST. Using `LogisticRegression` from sklearn, train a
# > baseline model on 2000 datapoints from the training set and report its
# > accuracy on the test set.

# %% [markdown]
# ### (b)
# > Using the `Linear` module from `dl.py`, train your own logistic regression
# > model on the MNIST dataset using gradient descent on cross-entropy loss. Run
# > for two epochs on minibatches of size 256, and ensure your model has at
# > least 90% accuracy on the test set.

# %% [markdown]
# ### (c)
# > Write a `ReLU` module that applies the ReLU activation. (Remember that ReLU
# > replaces negative inputs with zero and passes positive inputs through.)
# > Explain how you derived the backward pass.

# %% [markdown]
# ### (d)
# > Using `Linear` and `ReLU` modules, build some kind of multi-layer perceptron
# > and train it on MNIST. Ensure your model has at least 95% accuracy on the
# > test set.

# %% [markdown]
# ### (e)
# > **Bonus**: Still using only `numpy` and `iml/`, design a MLP that achieves
# > at least 98% test accuracy and takes less than 5 minutes to train on a
# > typical consumer CPU.


# %% [markdown]
# ## Part 2: Linear Autoencoders

# %% [markdown]
# ### (a)
# > Let's start by training the model `Linear(3, 3)` on `ulu.csv` with the
# > reconstruction loss objective. Run full-batch gradient descent on the raw
# > data. Show your loss curve over 50 iterations and determine, without doing
# > any additional calculations, if gradient descent seems to have a good rate
# > of convergence.

# %% [markdown]
# ### (b)
# > Consider the `ulu.csv` dataset. Make an autoencoder that encodes datapoints
# > as two-dimensional vectors. Using only `numpy` and `iml/`, minimize its
# > reconstruction loss (in squared Euclidean distance) using gradient descent.
# > Report the final reconstruction loss and confirm that it is close to the
# > theoretical minimum.

# %% [markdown]
# ### (c)
# > Now implement a `DropoutOne` module whose forward pass zeros out the first
# > dimension of each input vector with probability $0.5.$ What is the correct
# > way to implement a backwards pass for this module?

# %% [markdown]
# ### (d)
# > By incorporating `DropoutOne` into the forward pass, train an autoencoder in
# > such a way that the first dimension of the code reliably distinguishes the
# > foreground and background parts of `ulu.csv`. Ensure that your model
# > converges within 500 iterations of full-batch gradient descent. What is the
# > optimal expected reconstruction loss for this model?


# %% [markdown]
# ## Part 3: Bouba–Kiki

# %% [markdown]
# ### (a)
# > Load MobileNetV3-small with `torchvision.models.mobilenet_v3_small()`.
# > Briefly describe the idea of a convolutional linear layer, and count the
# > number of weights in this model.

# %% [markdown]
# ### (b)
# > Using MobileNetV3-small, separate the images in `bouba_kiki.npz` into a
# > bouba class and a kiki class. Confirm that your technique works by showing a
# > few examples of the classification. (I recommend upscaling the images to
# > $244 \times 244$ with `Resize` from `torchvision.transforms` before passing
# > them to the model.)

# %% [markdown]
# ### (c)
# > Fetch some images of frogs and airplanes from the CIFAR100 dataset. Using
# > your method from the previous problem, determine whether frogs or airplanes
# > are more kiki. Do you agree with the model's judgment?

# %% [markdown]
# ### (d)
# > **Bonus**: Can you identify the earliest layer of MobileNetV3 that reliably
# > distinguishes between the bouba and kiki examples in our dataset?


