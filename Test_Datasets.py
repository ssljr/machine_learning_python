import matplotlib.pyplot as plt
import mglearn.plots
import numpy as np
from sklearn.datasets import make_blobs
from sklearn.neighbors import KNeighborsClassifier


def make_forge():
    # a carefully hand-designed dataset lol
    X, y = make_blobs(centers=2, random_state=4, n_samples=30)
    y[np.array([7, 27])] = 0
    mask = np.ones(len(X), dtype=np.bool_)
    mask[np.array([0, 1, 5, 26])] = 0
    X, y = X[mask], y[mask]
    return X, y


new_X, new_y = make_forge()

"""
print(new_X.shape)

X_train, X_test, y_train, y_test = train_test_split(new_X, new_y, random_state=0)

clf = KNeighborsClassifier(n_neighbors=3)

clf.fit(X_train, y_train)
"""

fig, axes = plt.subplots(1, 3, figsize=(10, 3))

for n_neighbors, ax in zip([1, 3, 9], axes):
    clf = KNeighborsClassifier(n_neighbors=n_neighbors).fit(new_X, new_y)
    mglearn.plots.plot_2d_separator(clf, new_X, fill=True, eps=0.5, ax=ax, alpha=.4)
    mglearn.discrete_scatter(new_X[:, 0], new_X[:, 1], new_y, ax=ax)
    ax.set_title("{} neighbor(s)".format(n_neighbors))
    ax.set_xlabel("feature 0")
    ax.set_ylabel("feature 1")

axes[0].legend(loc=3)
plt.show()
