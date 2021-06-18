# nn-clustering-pytorch
Checking the divisibility of neural networks, and investigating the nature of the pieces networks can be divided into.

NOTE: this code is still under active development. Use at your own peril.

Things this codebase can do:

- Train a neural network using gradient descent and pruning. The neural network can have convolutions, fully-connected layers, batch norm, dropout, and max pooling, as long as it uses only ReLU activations. This is done in `src/train_model.py`.
- Turn such a neural network into a graph and apply graph clustering to it. This is done in `src/spectral_cluster_model.py`.
- Compare the clusterability of a model to that of random shuffles of the model's weights. This is done in `src/shuffle_and_cluster.py`.
- Regularize graph-clusterability during training, while normalizing weights. This is done in `src/clusterability_gradient.py` and `src/train_model.py`. Note: high values of the regularization weight might cause networks to lose connection between their inputs and outputs. I advise using values between 1e-4 and 1e-1.

After cloning, enter the pipenv by running `pipenv shell`. To install dependencies:

- if you don't want to contribute to development, just run `pipenv install`.
- if you do want to contribute to development, install dependencies by running `pipenv install -d`, then install pre-commit hooks for `yapf`, `isort`, and `flake8` by running `pre-commit install`.

You should also probably create directories where results will be saved: `datasets`, `models`, `training_runs`, `clustering_runs`, and `shuffle_clust_runs`.

This codebase uses `sacred` extensively, which you can read about [here](https://sacred.readthedocs.io/en/stable/).

NB: the codebase assumes that networks are defined in a very specific way. The operations must be grouped either into ModuleDicts (see [this pytorch doc page](https://pytorch.org/docs/stable/generated/torch.nn.ModuleDict.html)) or nn.Sequential modules.

For ModuleDicts:
- One ModuleDict should be used per layer.
- Batch norm modules must be in the ModuleDict of the linear transform they come after.
- Within a ModuleDict, fully connected modules should be have the key `'fc'`, convolutional modules should have the key `'conv'`, and batch norm modules should have the key `'bn'`. Nothing else should have those keys.
- ModuleDicts can't contain both a convolutional and a fully connected module.

For Sequential modules:
- Sequential modules should be formed by taking in an ordered dict, where keys are names of operations and values are modules.
- The keys of the ordered dict should be formatted `<name of layer>_<name of module>`, e.g. `0_fc`, `5_conv`, `layer6_relu`. Rules for names of modules are as for ModuleDicts.
- One layer should contain at most one fully-connected or convolutional module.

In both cases, there can only be one batch norm module between two linear transformations. For examples of networks that meet these specifications, see `src/networks.py`.
