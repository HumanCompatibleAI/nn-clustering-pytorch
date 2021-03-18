# nn-clustering-pytorch
Checking the divisibility of neural networks, and investigating the nature of the pieces networks can be divided into.

After cloning, enter the pipenv by running `pipenv shell`, install dependencies by running `pipenv install`, then install pre-commit hooks for `yapf`, `isort`, and `flake8` by running `pre-commit install`.

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
