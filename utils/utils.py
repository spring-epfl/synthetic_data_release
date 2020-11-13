import json
import random
from os import path


import numpy as np
from pandas import cut, get_dummies
from IPython.display import display_html


import logging
from logging.config import fileConfig
dirname = path.dirname(__file__)
logconfig = path.join(dirname, '../logging_config.ini')
fileConfig(logconfig)
logger = logging.getLogger()


def json_numpy_serialzer(o):
    """ Serialize numpy types for json

    Parameters:
        o (object): any python object which fails to be serialized by json

    Example:

        >>> import json
        >>> a = np.array([1, 2, 3])
        >>> json.dumps(a, default=json_numpy_serializer)

    """
    numpy_types = (
        np.bool_,
        np.float16,
        np.float32,
        np.float64,
        # np.float128,  -- special handling below
        np.int8,
        np.int16,
        np.int32,
        np.int64,
        np.str_,
        np.timedelta64,
        np.uint8,
        np.uint16,
        np.uint32,
        np.uint64,
        np.void,
    )

    if isinstance(o, np.ndarray):
        return o.tolist()
    elif isinstance(o, numpy_types):
        return o.item()
    elif isinstance(o, np.float128):
        return o.astype(np.float64).item()
    else:
        raise TypeError("{} of type {} is not JSON serializable".format(repr(o), type(o)))


def set_random_seed(seed=0):
    random.seed(seed)
    np.random.seed(seed)


def read_json_file(json_file):
    with open(json_file, 'r') as file:
        return json.load(file)


def preprocess_germancredit(df):
    # Create age categories
    interval = (18, 25, 35, 60, 120)
    cats = ['Student', 'Young', 'Adult', 'Senior']
    df["Age_cat"] = cut(df['Age'], interval, labels=cats)
    df.drop('Age', axis=1)

    # Impute missing values
    df['Saving accounts'] = df['Saving accounts'].fillna('no_inf')
    df['Checking account'] = df['Checking account'].fillna('no_inf')

    # Normalise credit amount values
    df['Credit amount'] = np.log(df['Credit amount'])

    # Dummy encode categorical variables
    for c in list(df.select_dtypes(['object', 'category'])):
        df = df.merge(get_dummies(df[c], drop_first=True, prefix=c.split(' ')[0]),
                                      left_index=True, right_index=True)
        df = df.drop(c, axis=1)

    return df


def preprocess_adult(df):
    # Create age categories
    interval = (18, 25, 35, 60, 120)
    cats = ['Student', 'Young', 'Adult', 'Senior']
    df["age_cat"] = cut(df['age'], interval, labels=cats)
    df.drop('age', axis=1)

    # Normalise capital gain and loss values
    # df['capital-gain'] = np.log(df['capital-gain'])
    # df['capital-loss'] = np.log(df['capital-loss'])

    # imputer_cat = SimpleImputer(strategy='most_frequent')
    cat_cols = list(df.select_dtypes(['object', 'category']))
    # imputer_cat.fit(df[cat_cols])
    # df[cat_cols] = imputer_cat.transform(df[cat_cols])
    #
    # imputer_num = SimpleImputer(strategy='median')
    # num_cols = list(df.select_dtypes(['float32']))
    # print(num_cols)
    # imputer_num.fit(df[num_cols])
    # df[num_cols] = imputer_num.transform(df[num_cols])

    # Dummy encode categorical variables
    for c in cat_cols:
        df = df.merge(get_dummies(df[c], drop_first=True, prefix=c.split(' ')[0]),
                      left_index=True, right_index=True)
        df = df.drop(c, axis=1)

    return df.dropna()

def encode_df(df, drop_first=True):
    df = df.copy()
    cat_cols = list(df.select_dtypes(['object', 'category']))

    for c in cat_cols:
        df = df.merge(get_dummies(df[c], drop_first=drop_first, prefix=c.split(' ')[0], prefix_sep=' '),
                        left_index=True, right_index=True)
        df = df.drop(c, axis=1)

    return df.dropna()


def display_side_by_side(*args):
    html_str=''
    for df in args:
        html_str+=df.to_html()
    display_html(html_str.replace('table','table style="display:inline"'), raw=True)





