# nn-clustering-pytorch
Checking the divisibility of neural networks, and investigating the nature of the pieces networks can be divided into.

Things this codebase can do:

- Train a neural network using gradient descent and pruning. The neural network can have convolutions, fully-connected layers, batch norm, dropout, and max pooling, as long as it uses only ReLU activations. This is done in `src/train_model.py`.
- Turn such a neural network into a graph and apply graph clustering to it. This is done in `src/spectral_cluster_model.py`.
- Compare the clusterability of a model to that of random shuffles of the model's weights. This is done in `src/shuffle_and_cluster.py`.
- Regularize graph-clusterability during training, while normalizing weights. This is done in `src/clusterability_gradient.py` and `src/train_model.py`. Note: high values of the regularization weight might cause networks to lose connection between their inputs and outputs.

After cloning, enter the pipenv by running `pipenv shell`. To install dependencies:

- if you don't want to contribute to development, just run `pipenv install`.
- if you do want to contribute to development, install dependencies by running `pipenv install -d`, then install pre-commit hooks for `yapf`, `isort`, and `flake8` by running `pre-commit install`.

This codebase uses `sacred` extensively, which you can read about [here](https://sacred.readthedocs.io/en/stable/).

NB: clustering requires the names of layers in networks to follow a pattern:

- fully connected layers should have names starting with `fc`
- convolutional layers should have names starting with `conv`
- batch norm layers should have names starting with `bn`

Also, the only layers to have names starting with `fc` should be fully connected layers, etc.

NB: the codebase assumes that networks are defined in a very specific way:

- Modules are grouped into ModuleDicts (see [this pytorch doc page](https://pytorch.org/docs/stable/generated/torch.nn.ModuleDict.html)), with one ModuleDict per layer.
- Names of ModuleDicts can't contain periods.
- Batch norm modules must be in the ModuleDict of the linear transform they come after.
- Within a ModuleDict, fully connected modules should be have the key `'fc'`, convolutional modules should have the key `'conv'`, and batch norm modules should have the key `'bn'`. Nothing else should have those keys.
- ModuleDicts can't contain both a convolutional and a fully connected module.
- There can only be one batch norm module between two linear transformations.

For examples of networks that meet these specifications, see `src/train_model.py`.

TODO: Write function to test whether network def complies with format.
TODO: use torch.autograd.gradcheck to see if I'm calculating clust grad right
