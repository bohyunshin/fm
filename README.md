# fm

Implementation of ctr / cvr models using pytorch or numpy (from scratch).

## Setting up environment

We use [poetry](https://github.com/python-poetry/poetry) to manage dependencies of repository.

Use poetry with version `2.1.1`.

```shell
$ poetry --version
Poetry (version 2.1.1)
```

Python version should be `3.11.x`.

```shell
$ python --version
Python 3.11.11
```

If python version is lower than `3.11`, try installing required version using `pyenv`.

Create virtual environment.

```shell
$ poetry env activate
```

If your global python version is not 3.11, run following command.

```shell
$ poetry env use python3.11
```

You can check virtual environment path info and its executable python path using following command.

```shell
$ poetry env info
```

After setting up python version, just run following command which will install all the required packages from `poetry.lock`.

```shell
$ poetry install
```

## Setting up git hook

Set up automatic linting using the following commands:
```shell
# This command will ensure linting runs automatically every time you commit code.
pre-commit install
```

## How to run

### Dataset

Currently, only criteo dataset is supported.

- For non-kaggle data, download it in [this link](https://ailab.criteo.com/criteo-sponsored-search-conversion-log-dataset/) before running experiment and specify file path for `CriteoSearchData`
- For kaggle data, download it in [this link](https://ailab.criteo.com/ressources/) and specify file path for `train.txt`.

### Quick Start

You can train implemented models using criteo dataset as

```bash
$ poetry run python3 src/train_torch.py \
	--data_path data/criteo/CriteoSearchData \
	--data_name criteo \
	--model fm \
	--is_test
```


## Experiment Results

|Model                         |Dataset  |Test loss|Test Macro F1|
|------------------------------|---------|---------|-------------|
| Logistic Regression          | criteo  | TBD     | TBD         |
| Factorization Machine        | criteo  | TBD     | TBD         |


## How to run pytest

Run following command.

```bash
$ poetry run pytest
```
