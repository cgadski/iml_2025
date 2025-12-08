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
# > Write a `ReLU` module that applies the ReLU activation. (Remember that ReLU
# > replaces negative inputs with zero and passes positive inputs through.)
# > Explain how you derived the backward pass.

# %% [markdown]
# ### (c)
# > Using `Linear` and `ReLU` modules, build some kind of multi-layer perceptron
# > and train it on MNIST. Ensure your model has at least 95% accuracy on the
# > test set.

# %% [markdown]
# ### (d)
# > **Bonus**: Still using only `numpy` and `iml/`, design a MLP that achieves
# > at least 98% test accuracy and trains in a short amount of time. (Say: under
# > 5 minutes on a median consumer CPU). You may consider using a library like
# > optuna.


# %% [markdown]
# ## Part 2: Linear Autoencoders

# %% [markdown]
# ### (a)
# > Let's start by training the model `Linear(3, 3)` on `ulu.csv` with the
# > reconstruction loss objective. Initialize the weight matrix randomly (this
# > is the default behavior for `Linear`) and run full-batch gradient descent
# > using the raw data as input for 50 iterations. Does there exist any learning
# > rate for which gradient descent has a good rate of convergence?

# %% [markdown]
# ### (b)
# > Make an autoencoder that encodes the datapoints of `ulu.csv` as
# > two-dimensional vectors. Minimize its reconstruction loss (in squared
# > Euclidean distance) using gradient descent. Report the final reconstruction
# > loss and confirm that it is close to the theoretical minimum.

# %% [markdown]
# ### (c)
# > Implement a `DropoutOne` module whose forward pass zeros out the first
# > dimension of each input vector with independent probability $0.5.$ What is
# > the correct way to implement a backwards pass for this module?

# %% [markdown]
# ### (d)
# > By incorporating `DropoutOne` into the forward pass, train an autoencoder in
# > such a way that the first dimension of the code reliably distinguishes the
# > foreground and background parts of `ulu.csv`. Ensure that your model
# > converges approximately within 500 iterations of full-batch gradient descent
# > and log its reconstruction loss. How can you determine the optimal expected
# > reconstruction loss for this model?


# %% [markdown]
# ## Part 3: Bouba–Kiki

# %% [markdown]
# ### (a)
# > Load
# > [mobilenet_v3_small](https://docs.pytorch.org/vision/main/models/generated/torchvision.models.mobilenet_v3_small.html)
# > using the `torchvision` package. Briefly describe the idea of a
# > convolutional linear layer. Identify one convolutional layer inside
# > MobileNetV3 and count how many parameters it has.

# %% [markdown]
# ### (b)
# > Without finetuning and without using the classification head, find a way to
# > use MobileNetV3-small to separate the images in `bouba_kiki.npz` into a
# > bouba class and a kiki class. Confirm that your technique works by showing a
# > few examples of the classification. (I recommend upscaling the images to
# > $224 \times 224$ with `Resize` from `torchvision.transforms` before passing
# > them to the model.)

# %% [markdown]
# ### (c)
# > Fetch some images of frogs and airplanes from the CIFAR10 dataset. Using
# > your method from the previous problem, determine whether frogs or airplanes
# > are more kiki. Do you agree with the model's judgment?

# %% [markdown]
# ### (d)
# > **Bonus**: Can you identify the earliest layer of MobileNetV3 that
# > distinguishes between the bouba and kiki examples in our dataset? Describe
# > your approach.


