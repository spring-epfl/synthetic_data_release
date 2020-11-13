# Privacy evaluation framework for synthetic data publishing
Implementation of a privacy evaluation framework for synthetic data publishing

# Attack models
The module `attack_models` so far includes:

- `MIAttackClassifier` is a privacy adversary that implements a generative model MIA, and can be used to evaluate the risk of linkability.
Given a single synthetic dataset output by a generative model, this adversary produces a binary label that predicts whether a target record belongs to the model’s training set or not


- `AttributeInferenceAttack` is a privacy adversary that learns to predict the value of an unknown sensitive attribute from a set of known attributes, and uses this knowledge to guess a target record’s sensitive value.

# Generative models
The module `generative_models` so far includes:   
- `IndependentHistogramModel`: An independent histogram model adapted from [Data Responsibly's DataSynthesiser](https://github.com/DataResponsibly/DataSynthesizer)
- `BayesianNetModel`: A generative model based on a Bayesian Network adapted from [Data Responsibly's DataSynthesiser](https://github.com/DataResponsibly/DataSynthesizer)
- `GaussianMixtureModel`: A simple Gaussian Mixture model taken from the [sklearn library](https://scikit-learn.org/stable/modules/generated/sklearn.mixture.GaussianMixture.html) 
- `CTGAN`: A conditional tabular generative adversarial network that integrates the CTGAN model from [CTGAN](https://github.com/sdv-dev/CTGAN)  
- `PateGan`: ﻿A model that builds on the Private Aggregation of Teacher Ensembles (PATE) to achieve differential privacy for GANs adapted from [PateGan](https://bitbucket.org/mvdschaar/mlforhealthlabpub/src/82d7f91d46db54d256ff4fc920d513499ddd2ab8/alg/pategan/)

# Setup
## Requirements
The framework and its building blocks have been developed and tested on Python 3.6 and 3.7

We recommend to create a virtual environment for installing all dependencies and running the code
```
python3 -m venv pyvenv3
source pyvenv3/bin/activate
pip install -r requirements.txt
```

## Dependencies

### PyTorch

The PyTorch package to install depends on the version of CUDA (if any) installed on your system. Please refer to their [website](https://pytorch.org/) to install the correct PyTorch package on your virtual environment.

### CGTAN

The `CTGAN` model depends on a fork of the original model training algorithm that can be found [here](https://github.com/spring-epfl/CTGAN)

To install the correct version clone the repository above and run
```
cd CTGAN
make install
```

To test your installation try to run 
```
import ctgan
```
from within your virtualenv `python`


## Unittests
To run the test suite included in `tests` run

```$xslt
python -m unittest discover
```

# Example
To run an example evaluation of the expected privacy gain with respect to the risk of linkability for all five generative models you can run
```$xslt
python mia_cli.py -D data/germancredit -RC runconfig_mia_example.json -O .
```

To run an example evaluation of the expected privacy gain with respect to the risk of attribute inference for all five generative models you can run
```$xslt
python mleai_cli.py -D data/germancredit -RC runconfig_attr_example.json -O .
```

