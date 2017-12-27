import argparse
import utils
import pandas as pd

from sklearn import manifold
from sklearn.decomposition import PCA


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--transformation', '-t')
    parser.add_argument('--n_components', '-n')
    parser.add_argument('--output', '-o')
    parser.add_argument('--data',  '-d')

    return parser.parse_args()


def get_transformation(transformation, n_components):
    return {
        "pca": PCA(n_components=n_components),
        "tsne": manifold.TSNE(n_components=n_components,
                              init='pca', random_state=0)
    }.get(transformation)


def main():
    args = get_args()

    data = utils.load_data(args.data)
    X = utils.get_features(data)

    transformation = get_transformation(args.transformation, arg.n_components)

    with open(args.output, 'w') as f:
        output = transformation.fit_transform(X)
        output.to_csv(f, sep=",")


if __name__ == '__main__':
    main()
