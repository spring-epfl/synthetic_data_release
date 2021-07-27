#!/usr/bin/env python3
"""
Launcher to process the PrivBayes mechanism.

This script is an adaptation of the execution scripts from
https://github.com/spring-epfl/synthetic_data_release.

-----
Nampoina Andriamilanto <tompo.andri@gmail.com>
"""

from argparse import ArgumentParser
from ast import literal_eval
from pathlib import Path
from warnings import simplefilter
simplefilter('ignore', category=FutureWarning)
simplefilter('ignore', category=DeprecationWarning)

from loguru import logger

from generative_models.ctgan import CTGAN
from generative_models.data_synthesiser import (
    IndependentHistogram, BayesianNet, PrivBayes)
from generative_models.pate_gan import PATEGAN
from utils.datagen import load_s3_data_as_df, load_local_data_as_df


DEFAULT_SAMPLE_SIZE = 1000


def main():
    """Execute the PrivBayes mechanism."""
    # Parse the arguments
    argparser = ArgumentParser()
    datasource = argparser.add_mutually_exclusive_group()
    datasource.add_argument('--s3name', '-S3', type=str, choices=[
                            'adult', 'census', 'credit', 'alarm', 'insurance'],
                            help='Name of the dataset to run on')
    datasource.add_argument('--datapath', '-D', type=str,
                            help='Path to a local data file')
    argparser.add_argument('--mechanism', '-M', type=str, choices=[
        'IndependentHistogram', 'BayesianNet', 'PrivBayes', 'CTGAN', 'PATEGAN'
        ], default='PrivBayes', help='The mechanism to use')
    argparser.add_argument('--parameters', '-P', type=str, default=None,
                           help='The parameters of the mechanism to use '
                                'separated by a colon')
    argparser.add_argument('--output-file', '-O', type=str,
                           help='The file where to store the synthetic dataset'
                           )
    argparser.add_argument('--sample-size', '-N', type=int,
                           default=DEFAULT_SAMPLE_SIZE,
                           help='The size of the synthetic dataset')
    args = argparser.parse_args()

    # Load data
    if args.s3name:
        raw_pop, metadata = load_s3_data_as_df(args.s3name)
        dname = args.s3name
    elif args.datapath:
        raw_pop, metadata = load_local_data_as_df(Path(args.datapath))
        dname = args.datapath.split('/')[-1]
    else:
        raise ValueError('Please provide a dataset')
    logger.info(f'Loaded data {dname}:\n{raw_pop}')
    logger.info(f'Loaded the corresponding metadata: {metadata}')

    # Initialize the mechanism
    parameters = []
    if args.parameters:
        parameters = [literal_eval(param)
                      for param in args.parameters.split(',')]
    logger.debug(f'Parameters: {parameters}')

    # IndependentHistogram parameters:
    # histogram_bins=10, infer_ranges=False, multiprocess=True
    if args.mechanism == 'IndependentHistogram':
        mechanism = IndependentHistogram(metadata, *parameters)

    # BayesianNet parameters:
    # histogram_bins=10, degree=1, infer_ranges=False, multiprocess=True,
    # seed=None
    elif args.mechanism == 'BayesianNet':
        mechanism = BayesianNet(metadata, *parameters)

    # PrivBayes parameters:
    # histogram_bins=10, degree=1, epsilon=.1, infer_ranges=False,
    # multiprocess=True, seed=None
    elif args.mechanism == 'PrivBayes':
        mechanism = PrivBayes(metadata, *parameters)

    # CTGAN parameters:
    # embedding_dim=128, gen_dim=(256, 256), dis_dim=(256, 256), l2scale=1e-6,
    # batch_size=500, epochs=300, multiprocess=False
    elif args.mechanism == 'CTGAN':
        mechanism = CTGAN(metadata, *parameters)

    # PATEGAN parameters:
    # eps=1, delta=1e-5, infer_ranges=False, num_teachers=10, n_iters=100,
    # batch_size=128, learning_rate=1e-4, multiprocess=False
    elif args.mechanism == 'PATEGAN':
        mechanism = PATEGAN(metadata, *parameters)

    # Unknown mechanism
    else:
        raise ValueError(f'Unknown mechanism {args.mechanism}')

    # Set the output path
    output_path = Path(f'{mechanism.__name__}.csv')
    if args.output_file:
        output_path = Path(args.output_file)

    # Generate the synthetic data
    logger.info('Generating the synthetic data, this can take time...')
    mechanism.fit(raw_pop)
    mechanism.generate_samples(args.sample_size).to_csv(output_path)


if __name__ == "__main__":
    main()
