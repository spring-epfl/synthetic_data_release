#!/usr/bin/env python3
"""
Generate the json metadata file given a dataset in csv format.

Please set the two global variables IMPLICIT_ORDINAL_ATTRIBUTES and
EXPLICIT_ORDINAL_ATTRIBUTES to correspond to the dataset that you use.

Great care should be taken when using this script to infer the type and the
domain of the attributes as it relies on the dataset that is given in
parameter.


usage: generate_metadata_file.py [-h] --dataset DATASET [--output OUTPUT]

optional arguments:
  -h, --help            show this help message and exit
  --dataset DATASET, -i DATASET
                        Path to the dataset in csv format
  --output OUTPUT, -o OUTPUT
                        Path where to write the json metadata file

-----
Nampoina Andriamilanto <tompo.andri@gmail.com>
"""

import json
from argparse import ArgumentParser
from pathlib import Path
from typing import Set

import numpy as np
import pandas as pd
from loguru import logger

from utils.constants import (CATEGORICAL, FLOAT, INTEGER, NUMERICAL, ORDINAL)

# Please define the set of the ordinal attributes which values can be
# automatically sorted (using the sorted() python function)
IMPLICIT_ORDINAL_ATTRIBUTES = {'age', 'fnlwgt'}

# Please define the set of the ordinal attributes which values are ordered
# manually
EXPLICIT_ORDINAL_ATTRIBUTES = {
    'education': ['Preschool', '1st-4th', '5th-6th', '7th-8th', '9th', '10th',
                  '11th', '12th', 'HS-grad', 'Prof-school', 'Assoc-acdm',
                  'Assoc-voc', 'Some-college', 'Bachelors', 'Masters',
                  'Doctorate']}

ORDINAL_ATTRIBUTES = IMPLICIT_ORDINAL_ATTRIBUTES.union(
    set(EXPLICIT_ORDINAL_ATTRIBUTES.keys()))

OUTPUT_FILE_SUFFIX = '.json'
JSON_SPACE_INDENT = 2


def main():
    """Generate the json metadata file."""
    # Parse the arguments
    argparser = ArgumentParser()
    argparser.add_argument('--dataset', '-i', type=str, required=True,
                           help='Path to the dataset in csv format')
    argparser.add_argument('--output', '-o', type=str,
                           help='Path where to write the json metadata file')
    args = argparser.parse_args()

    # Load the dataset
    logger.info(f'Loading the data from {args.dataset}')
    dataset_path = Path(args.dataset)
    dataset = pd.read_csv(dataset_path, header=0)
    logger.debug(f'Sample of the loaded dataset:\n{dataset}')

    # Generate the metadata of each attribute
    logger.info('Generating the metadata of the attributes')
    attributes = []
    for column in dataset.columns:
        # Get the numpy type of the column
        numpy_type = dataset[column].dtype
        logger.debug(f'{column} has the numpy type {numpy_type}')

        # Infer its type among (Integer, Float, Ordinal, Categorical)
        inferred_type = infer_type(column, numpy_type, ORDINAL_ATTRIBUTES)
        column_infos = {'name': column, 'type': inferred_type}

        # If the type is numerical, set the min and max value
        if inferred_type in NUMERICAL:
            column_infos['min'] = dataset[column].min()
            column_infos['max'] = dataset[column].max()
        else:
            # If the type is explicitely ordinal, we retrieve its ordered
            # values which are set manually in EXPLICIT_ORDINAL_ATTRIBUTES.
            # Otherwise (implicit ordinal or categorical), we get the sorted
            # list of values from the dataset (the second parameter of get()).
            ordered_values = EXPLICIT_ORDINAL_ATTRIBUTES.get(
                column, sorted(dataset[column].unique()))
            column_infos['size'] = len(ordered_values)

            # If the values are numbers, we cast them to strings as the
            # metadata configuration files seem to have the values of ordinal
            # and categorical attributes specified as strings
            if isinstance(ordered_values[0], np.number):
                ordered_values = [str(value) for value in ordered_values]

            column_infos['i2s'] = ordered_values

        attributes.append(column_infos)

    # Write the json metadata file
    if args.output:
        output_path = args.output
    else:
        output_path = dataset_path.with_name(
            dataset_path.stem + OUTPUT_FILE_SUFFIX)
    logger.info(f'Writting the metadata to {output_path}')

    with open(output_path, 'w+') as json_output_file:
        json.dump({'columns': attributes}, json_output_file,
                  indent=JSON_SPACE_INDENT)


def infer_type(column: str, numpy_type: str, ordinal_attributes: Set[str]
               ) -> str:
    """Infer the type of an attribute given its numpy type.

    Args:
        column: The name of the column.
        numpy_type: The numpy type of the column.
        ordinal_attributes: The set of the ordinal attributes.
    """
    if isinstance(numpy_type, np.integer):
        return INTEGER
    if isinstance(numpy_type, np.floating):
        return FLOAT
    if column in ordinal_attributes:
        return ORDINAL
    return CATEGORICAL


if __name__ == "__main__":
    main()
