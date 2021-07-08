import json
import random

import numpy as np
import multiprocessing as mp

from warnings import simplefilter
simplefilter('ignore', category=FutureWarning)
simplefilter('ignore', category=DeprecationWarning)


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


def get_mia_gain(pCorrectSyn):
    # return min(1, 2*(1 - pCorrectSyn))
    return 2 * (1 - pCorrectSyn)


def get_accuracy(guesses, labels):
    return sum([g == l for g, l in zip(guesses, labels)])/len(labels)


class CustomProcess(mp.Process):
    def run(self, *args, **kwargs):
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', category=FutureWarning)
            warnings.simplefilter('ignore', category=DeprecationWarning)
            return mp.Process.run(self, *args, **kwargs)






