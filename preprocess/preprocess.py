import pandas as import pd

from sklearn import manifold


def tsne(data):
    tsne = manifold.TSNE(n_components=n_components, init='pca', random_state=0)
    return tsne.fit_transform(data)
