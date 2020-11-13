"""
Helper functions for loading, converting, reshaping data
"""

import pandas as pd
import json
from pandas.api.types import CategoricalDtype

FILLNA_VALUE_CAT = 'NaN'
CATEGORICAL = "categorical"
CONTINUOUS = "continuous"
ORDINAL = "ordinal"


def load_local_data_as_df(filename):
    with open(f'{filename}.json') as f:
        metadata = json.load(f)
    dtypes = {cd['name']:_get_dtype(cd) for cd in metadata['columns']}
    df = pd.read_csv(f'{filename}.csv', dtype=dtypes)
    metadata['categorical_columns'], metadata['ordinal_columns'], metadata['continuous_columns'] = _get_columns(metadata)

    return df, metadata


def load_local_data_as_array(filename):
    df = pd.read_csv(f'{filename}.csv')
    with open(f'{filename}.json') as f:
        metadata = json.load(f)
    metadata['categorical_columns'], metadata['ordinal_columns'], metadata['continuous_columns'] = _get_columns(metadata)

    data = convert_df_to_array(df, metadata)

    return data, metadata


def _get_dtype(cd):
    if cd['type'] == 'continuous':
        return float
    else:
        return str


def _get_columns(metadata):
    categorical_columns = list()
    ordinal_columns = list()
    continuous_columns = list()
    for column_idx, column in enumerate(metadata['columns']):
        if column['type'] == CATEGORICAL:
            categorical_columns.append(column_idx)
        elif column['type'] == ORDINAL:
            ordinal_columns.append(column_idx)
        elif column['type'] == CONTINUOUS:
            continuous_columns.append(column_idx)

    return categorical_columns, ordinal_columns, continuous_columns


def _load_json(path):
    with open(path) as json_file:
        return json.load(json_file)


def convert_array_to_df(data, metadata):
    df = pd.DataFrame(data)
    column_names = []
    for i, col in enumerate(metadata['columns']):
        column_names.append(col['name'])
        if col['type'] in [CATEGORICAL, ORDINAL]:
            df.iloc[:, i] = df.iloc[:, i].astype('object')
            df.iloc[:, i] = df.iloc[:, i].map(pd.Series(col['i2s']))

    df.columns = column_names
    return df


def convert_df_to_array(df, metadata):
    dfcopy = df.copy()
    for col in metadata['columns']:
        if col['name'] in list(dfcopy):
            col_data = dfcopy[col['name']]
            if col['type'] in [CATEGORICAL, ORDINAL]:
                if len(col_data) > len(col_data.dropna()):
                    col_data = col_data.fillna(FILLNA_VALUE_CAT)
                    if FILLNA_VALUE_CAT not in col['i2s']:
                        col['i2s'].append(FILLNA_VALUE_CAT)
                        col['size'] += 1
                cat = CategoricalDtype(categories=col['i2s'], ordered=True)
                col_data = col_data.astype(cat)
                dfcopy[col['name']] = col_data.cat.codes

    return dfcopy.values


def convert_series_to_array(scopy, metadata):
    scopy = scopy.copy()
    for col in metadata['columns']:
        if col['name'] == scopy.name:
            if col['type'] in [CATEGORICAL, ORDINAL]:
                if len(scopy) > len(scopy.dropna()):
                    scopy = scopy.fillna(FILLNA_VALUE_CAT)
                    if FILLNA_VALUE_CAT not in col['i2s']:
                        col['i2s'].append(FILLNA_VALUE_CAT)
                        col['size'] += 1
                cat = CategoricalDtype(categories=col['i2s'], ordered=True)
                scopy = scopy.astype(cat)
                scopy = scopy.cat.codes

    return scopy.values


