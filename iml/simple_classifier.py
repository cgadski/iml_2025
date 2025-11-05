from matplotlib.colors import LogNorm, BoundaryNorm
from sklearn.base import BaseEstimator, ClassifierMixin
import matplotlib.pyplot as plt
import numpy as np


class SimpleClassifier(ClassifierMixin, BaseEstimator):
    def __init__(self, n_classes: int):
        self.n_classes = n_classes

    def predict_proba(self, X):
        raise NotImplementedError()

    def predict(self, x):
        probs = self.predict_proba(x)
        return np.argmax(probs, axis=1)

    def fit(self, x, y, lr, steps=20):
        raise NotImplementedError()

    def show(self, x, y):
        _, ax = plt.subplots()
        show_classifier(ax, x, y, self)


def show_classifier(ax, x, y, model):
    """
    Plot some two-dimensional data and labels overlaid with the predictions of a classification model.
    """
    k = y.max()
    x_min, x_max = x.min(axis=0), x.max(axis=0)

    res = 100

    x0, x1 = np.meshgrid(
        np.linspace(x_min[0], x_max[0], num=res),
        np.linspace(x_min[1], x_max[1], num=res),
    )

    x_mesh = np.stack([x0, x1], axis=-1).reshape(-1, 2)
    probs = model.predict_proba(x_mesh).reshape(res, res, -1)
    classes = np.argmax(probs, axis=-1)
    predicted_prob = probs[np.arange(res)[:, None], np.arange(res), classes]
    cmap = plt.get_cmap("tab10")
    ax.pcolormesh(
        x0,
        x1,
        classes,
        cmap=cmap,
        norm=BoundaryNorm(np.arange(k + 1), k),
    )

    ax.scatter(
        x[:, 0],
        x[:, 1],
        c=cmap(y),
        edgecolors="white",
    )
