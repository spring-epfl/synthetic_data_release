# Privacy evaluation framework for synthetic data publishing
Implementation of a privacy evaluation framework for synthetic data publishing

Based on "Synthetic Data - Anonymisation Groundhog Day, Theresa Stadler, Bristena Oprisanu, and Carmela Troncoso, [arXiv](https://arxiv.org/abs/2011.07018), 2020"

# Attack models
The module `attack_models` so far includes

A privacy adversary to test for privacy gain with respect to linkage attacks modelled as a membership inference attack `MIAAttackClassifier`.

A simple attribute inference attack `AttributeInferenceAttack` that aims to infer a target's sensitive value given partial knowledge about the target record

# Generative models
The module `generative_models` so far includes:   
- `IndependentHistogramModel`: An independent histogram model adapted from [Data Responsibly's DataSynthesiser](https://github.com/DataResponsibly/DataSynthesizer)
- `BayesianNetModel`: A generative model based on a Bayesian Network adapted from [Data Responsibly's DataSynthesiser](https://github.com/DataResponsibly/DataSynthesizer)
- `GaussianMixtureModel`: A simple Gaussian Mixture model taken from the [sklearn library](https://scikit-learn.org/stable/modules/generated/sklearn.mixture.GaussianMixture.html)
- `CTGAN`: A conditional tabular generative adversarial network that integrates the CTGAN model from [CTGAN](https://github.com/sdv-dev/CTGAN)  
- `PATE-GAN`: A differentially private generative adversarial network adapted from its original [implementation](https://bitbucket.org/mvdschaar/mlforhealthlabpub/src/82d7f91d46db54d256ff4fc920d513499ddd2ab8/alg/pategan/)

# Setup
## Requirements
The framework and its building blocks have been developed and tested under Python 3.6 and 3.7

We recommend to create a virtual environment for installing all dependencies and running the code
```
python3 -m venv pyvenv3
source pyvenv3/bin/activate
pip install -r requirements.txt
```

## Dependencies
The `CTGAN` model depends on a fork of the original model training algorithm that can be found here
[CTGAN-SPRING](https://github.com/spring-epfl/CTGAN.git)

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

# Example runs
To run a privacy evaluation with respect to the privacy concern of linkability you can run

```
python linkage_cli.py -D data/texas -RC tests/linkage/runconfig.json -O tests/linkage
```

The results file produced after successfully running the script can be parsed with the function `load_results_mia` provided in `utils/analyse_results.py`. 

To run a privacy evaluation with respect to the privacy concern of inference you can run

```
python inference_cli.py -D data/texas -RC tests/inference/runconfig.json -O tests/inference
```

The results file produced after successfully running the script can be parsed with the function `load_results_ai` provided in `utils/analyse_results.py`.

