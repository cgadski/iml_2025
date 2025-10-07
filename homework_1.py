# %%
%load_ext autoreload
%autoreload 2

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# %% [markdown]
# **Group members:** [your names]

# %% [markdown]
# ## Part 1: Regression

# %% [markdown]
# ### (a)
# > For the linear model $\hat y = \theta_0 + \theta_1 x,$ give a formula for
# > the partial derivatives of mean squared error with respect to $\theta_0$ and
# > $\theta_1.$ Give a general formula for the optimal parameters (should they
# > exist). Find optimal parameters for the provided dataset and graph the fit.

# %% [markdown]
# ### (b)
# > Let $L(\theta)$ be the mean squared error of our linear model with parameter
# > $\theta = (\theta_0, \theta_1).$ Putting $\theta = (0, 0),$ graph the
# > function $L(\theta - \eta \nabla L(\theta))$ as a function of $\eta$ in some
# > neighborhood of $0.$ Choose a small enough range to show that $- \nabla L$
# > is a descent direction.

# %% [markdown]
# ### (c)
# > Normalize the powerplant data in some way and fit a linear model using
# > gradient descent on mean squared error. Visualize how the model and the
# > error change during training. (Consider using code from
# > `iml/simple_regression.py`.)

# %% [markdown]
# ### (d)
# > Now try running gradient descent without first normalizing the power plant
# > data. Explain why the rate of convergence is slower.

# %% [markdown]
# ### (e)
# > Fit a polynomial model $y = \theta_0 + \theta_1 x + \theta_2 x^2$ using any
# > method you want and graph it. Explain whether fitting this model is possible
# > without using an iterative method like gradient descent.


# %% [markdown]
# ## Part 2: Classification

# %% [markdown]
# ### (a)
# > Give a formula for the partial derivatives of cross-entropy loss with
# > respect to the parameters $(W, b)$ of a multi-class logistic regression
# > problem.

# %% [markdown]
# ### (b)
# > Write a class, roughly with the API of scikit-learn's `LogisticRegression`,
# > that fits a multi-class logistic regression model using gradient descent on
# > the cross-entropy loss. Train your model on the iris dataset and report
# > cross-entropy loss and accuracy. (Consider normalizing your data.)

# %% [markdown]
# ### (c)
# > Train your logistic regression model with only sepal length and sepal width
# > as input features. Make a scatter plot of these two attributes overlaid with
# > the decision boundaries of your model. (Consider using code from
# > `iml/simple_classifier.py`.)

# %% [markdown]
# ### (d)
# > Consider scikit-learn's implementation of $k$-nearest neighbors and decision
# > tree. Using default hyperparameters, train these two models with sepal
# > length and sepal width as input features. Show their decision boundaries.

# %% [markdown]
# ### (e)
# > Of the three models you used, which would you choose to predict species as a
# > function of sepal length/width? Support your conclusion with some validation
# > metric.


# %% [markdown]
# ## Part 3: Unsupervised Methods

# %% [markdown]
# ### (a)
# > Suppose we didn't have access to the species attribute of the iris dataset.
# > Implement the Lloyd-Forgy algorithm for $k$-means and use it to infer a
# > partition of the iris dataset into three species. Compute the confusion
# > matrix and the accuracy of the partition relative to the true species.

# %% [markdown]
# ### (b)
# > Consider a vector-valued random variable $X \in \mathbb R^n$ and a vector
# > $\theta \in \mathbb R^n.$ Subject to the constraint
# > $\lVert \theta \rVert = 1,$ when is $\theta$ a strict maximizer for the
# > variance of the inner product $\langle \theta, X \rangle$? Derive a
# > characterization in terms of the covariance matrix of $X$ and illustrate
# > this problem with a two-dimensional dataset of your choosing.

# %% [markdown]
# ### (c)
# > Investigate `data/ulu.csv` using PCA and explain how this data was produced.
# > Using scikit-learn, run both $k$-means clustering and dbscan. Which method
# > produces more reasonable ``clusters"?

# %% [markdown]
# ### (d)
# > `data/low_rank.csv` contains 100 datapoints in 20 dimensions. You can
# > imagine this is sensor data from some physical system sampled over a short
# > period of time. Using PCA, produce a better estimate of the quantity
# > measured by the first sensor. Graph the raw sensor value and your denoised
# > estimate as a function of time. Briefly explain your approach.


# %% [markdown]
# ## Challenge Problem: Retail Dataset

# %% [markdown]
# ### (a)
# > Explore the online retail dataset. Train a simple model on some aspect of
# > the data and explain what it tells us. (It should help to put some effort
# > into your preprocessing/exploratory analysis!)


