# nn-clustering-pytorch
Checking the divisibility of neural networks, and investigating the nature of the pieces networks can be divided into.

Things this codebase can do:

- Train a neural network using gradient descent and pruning. The neural network can have convolutions, fully-connected layers, batch norm, dropout, and max pooling, as long as it uses only ReLU activations. This is done in `src/train_model.py`.
- Turn such a neural network into a graph and apply graph clustering to it. This is done in `src/spectral_cluster_model.py`.
- Compare the clusterability of a model to that of random shuffles of the model's weights. This is done in `src/shuffle_and_cluster.py`.
- Regularize graph-clusterability during training, while normalizing weights. This is done in `src/clusterability_gradient.py` and `src/train_model.py`.

After cloning, enter the pipenv by running `pipenv shell`. To install dependencies:

- if you don't want to contribute to development, just run `pipenv install`.
- if you do want to contribute to development, install dependencies by running `pipenv install -d`, then install pre-commit hooks for `yapf`, `isort`, and `flake8` by running `pre-commit install`.

This codebase uses `sacred` extensively, which you can read about [here](https://sacred.readthedocs.io/en/stable/).

NB: clustering requires the names of layers in networks to follow a pattern:

- fully connected layers should have names starting with `fc`
- convolutional layers should have names starting with `conv`
- batch norm layers should have names starting with `bn`

Also, the only layers to have names starting with `fc` should be fully connected layers, etc.

- Gotta group things up into named module dicts
- names can't contain periods
- batch norm has got to be in the module dict of the linear transform it comes after
- inside layers, names have to be exactly right
- layers can't contain both a conv and an fc layer
