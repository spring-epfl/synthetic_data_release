# Privacy evaluation framework for synthetic data publishing
A practical framework to evaluate the privacy-utility tradeoff of synthetic data publishing 

Based on "Synthetic Data - Anonymisation Groundhog Day, Theresa Stadler, Bristena Oprisanu, and Carmela Troncoso, [arXiv](https://arxiv.org/abs/2011.07018), 2020"

# Attack models
The module `attack_models` so far includes

A privacy adversary to test for privacy gain with respect to linkage attacks modelled as a membership inference attack `MIAAttackClassifier`.

A simple attribute inference attack `AttributeInferenceAttack` that aims to infer a target's sensitive value given partial knowledge about the target record

# Generative models
The module `generative_models` so far includes:   
- `IndependentHistogram`: An independent histogram model adapted from [Data Responsibly's DataSynthesiser](https://github.com/DataResponsibly/DataSynthesizer)
- `BayesianNet`: A generative model based on a Bayesian Network adapted from [Data Responsibly's DataSynthesiser](https://github.com/DataResponsibly/DataSynthesizer)
- `PrivBayes`: A differentially private version of the BayesianNet model adapted from [Data Responsibly's DataSynthesiser](https://github.com/DataResponsibly/DataSynthesizer)
- `CTGAN`: A conditional tabular generative adversarial network that integrates the CTGAN model from [CTGAN](https://github.com/sdv-dev/CTGAN)  
- `PATE-GAN`: A differentially private generative adversarial network adapted from its original implementation by the [MLforHealth Lab](https://bitbucket.org/mvdschaar/mlforhealthlabpub/src/82d7f91d46db54d256ff4fc920d513499ddd2ab8/alg/pategan/)

# Setup

## Docker Distribution

For your convenience, Synthetic Data is also distributed as a ready-to-use Docker image containing Python 3.9 and CUDA 11.4.2, along with all dependencies required by Synthetic Data, including jupyter notebook to visualise and analyse the results.

**Note:** This distribution includes CUDA binaries, before downloading the image, ensure to read [its EULA](https://docs.nvidia.com/cuda/eula/index.html) and to agree to its terms.

Pull the image and run a container (and bind a volume where you want to save the data):

```
docker pull springepfl/synthetic-data:latest
docker run -it --rm -v "$(pwd)/output:/output" -p 8888:8888 springepfl/synthetic-data
```

The Synthetic Data directory is placed at the root directory of the container.
```
cd /synthetic_data_release
```

You should now be able to run the examples without encountering any problems, and you should be able to visualize the results with Jupyter by running
```
jupyter notebook --allow-root --ip=0.0.0.0
```

and opening the notebook with your favourite web browser at the url `http://127.0.0.1:8888/?token=<authentication token>`.


## Direct Installation

### Requirements
The framework and its building blocks have been developed and tested under Python 3.9 .

We recommend to create a virtual environment for installing all dependencies and running the code
```
python3 -m venv pyvenv3
source pyvenv3/bin/activate
pip install numpy==1.19.5 && pip install -r requirements.txt
```

Note: Some people encountered problems due to the API of Numpy having changed between versions, to ensure all dependencies are compiled against the same Numpy version, it needs to be installed first.

### Dependencies
The `CTGAN` model depends on a fork of the original model training algorithm that can be found here
[CTGAN-SPRING](https://github.com/spring-epfl/CTGAN.git)

To install the correct version clone the repository above and run
```
cd CTGAN
make install
```

Add the path to this directory to your python path. You can also add this line
in your shell configuration file (e.g., `~/.bashrc`) to load it automatically.
```bash
# Execute this in the CTGAN folder, otherwise replace `pwd` with the actual path
export PYTHONPATH=$PYTHONPATH:`pwd`
```

To test your installation try to run
```
import ctgan
```
from within your virtualenv `python`

# Example runs
To run a privacy evaluation with respect to the privacy concern of linkability you can run

```
python3 linkage_cli.py -D data/texas -RC tests/linkage/runconfig.json -O tests/linkage
```

The results file produced after successfully running the script will be written to `tests/linkage` and can be parsed with the function `load_results_linkage` provided in `utils/analyse_results.py`. 
A jupyter notebook to visualise and analyse the results is included at `notebooks/Analyse Results.ipynb`.


To run a privacy evaluation with respect to the privacy concern of inference you can run

```
python3 inference_cli.py -D data/texas -RC tests/inference/runconfig.json -O tests/inference
```

The results file produced after successfully running the script can be parsed with the function `load_results_inference` provided in `utils/analyse_results.py`.
A jupyter notebook to visualise and analyse the results is included at `notebooks/Analyse Results.ipynb`.


To run a utility evaluation with respect to a simple classification task as utility function run

```
python3 utility_cli.py -D data/texas -RC tests/utility/runconfig.json -O tests/utility
```

The results file produced after successfully running the script can be parsed with the function `load_results_utility` provided in `utils/analyse_results.py`.

