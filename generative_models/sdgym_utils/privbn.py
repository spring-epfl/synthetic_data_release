import logging
import os
import pathlib
import shutil
import subprocess
import tempfile
import rdt
from datetime import datetime

import numpy as np
import pandas as pd

from sdgym.errors import UnsupportedDataset
from sdgym.constants import CATEGORICAL, ORDINAL
from sdgym.synthesizers.base import LegacySingleTableBaseline
from sdgym.synthesizers.utils import Transformer

cwd = os.path.dirname(__file__)
PRIVBAYES_BN = os.path.join(cwd, 'privbayes/privBayes.bin')
LOGGER = logging.getLogger(__name__)


def try_mkdirs(dir):
    if not os.path.isdir(dir):
        os.makedirs(dir)


class PrivBN(LegacySingleTableBaseline):
    """docstring for PrivBN."""

    def __init__(self, epsilon=1, theta=20, max_samples=25000):
        self.privbayes_bin = PRIVBAYES_BN
        if not os.path.exists(self.privbayes_bin):
            raise RuntimeError('privbayes binary not found. Please set PRIVBAYES_BIN')

        self.epsilon = epsilon
        self.theta = theta
        self.max_samples = max_samples

        self.columns = []
        self.transformed_columns = []

        self.ht = None
        self.model_data = None
        self.meta = None

    def fit(self, real_data, table_metadata):
        LOGGER.info("Fitting %s", self.__class__.__name__)
        self.columns, categoricals = self._get_columns(real_data, table_metadata)
        real_data = real_data[self.columns]

        self.ht = rdt.HyperTransformer(default_data_type_transformers={
            'categorical': 'LabelEncodingTransformer',
        })
        self.ht.fit(real_data.iloc[:, categoricals])
        model_data = self.ht.transform(real_data)

        supported = set(model_data.select_dtypes(('number', 'bool')).columns)
        unsupported = set(model_data.columns) - supported
        if unsupported:
            unsupported_dtypes = model_data[unsupported].dtypes.unique().tolist()
            raise UnsupportedDataset(f'Unsupported dtypes {unsupported_dtypes}')

        nulls = model_data.isnull().any()
        if nulls.any():
            unsupported_columns = nulls[nulls].index.tolist()
            raise UnsupportedDataset(f'Null values found in columns {unsupported_columns}')

        self.transformed_columns = list(model_data)
        self.model_data = model_data.to_numpy().copy()
        self.meta = Transformer.get_metadata(self.model_data, categoricals, ())

    def sample(self, n):
        LOGGER.info("Sampling %s", self.__class__.__name__)

        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = pathlib.Path(tmpdir)
            try_mkdirs(tmpdir / 'data')
            try_mkdirs(tmpdir / 'log')
            try_mkdirs(tmpdir / 'output')
            shutil.copy(self.privbayes_bin, tmpdir / 'privBayes.bin')
            d_cols = []
            with open(tmpdir / 'data/real.domain', 'w') as f:
                for id_, info in enumerate(self.meta):
                    if info['type'] in [CATEGORICAL, ORDINAL]:
                        print('D', end='', file=f)
                        counter = 0
                        for i in range(info['size']):
                            if i > 0 and i % 4 == 0:
                                counter += 1
                                print(' {', end='', file=f)
                            print('', i, end='', file=f)
                        print(' }' * counter, file=f)
                        d_cols.append(id_)
                    else:
                        minn = info['min']
                        maxx = info['max']
                        d = (maxx - minn) * 0.03
                        minn = minn - d
                        maxx = maxx + d
                        print('C', minn, maxx, file=f)

            with open(tmpdir / 'data/real.dat', 'w') as f:
                np.random.shuffle(self.model_data)
                n = min(n, self.max_samples)
                for i in range(n):
                    row = self.model_data[i]
                    for id_, col in enumerate(row):
                        if id_ in d_cols:
                            print(int(col), end=' ', file=f)

                        else:
                            print(col, end=' ', file=f)

                    print(file=f)

            privbayes = os.path.realpath(tmpdir / 'privBayes.bin')
            arguments = [privbayes, 'real', str(n), '1', str(self.epsilon), str(self.theta)]
            LOGGER.info('Calling %s', ' '.join(arguments))
            start = datetime.utcnow()
            subprocess.call(arguments, cwd=tmpdir)
            LOGGER.info('Elapsed %s', datetime.utcnow() - start)

            sampled_data = np.loadtxt(tmpdir / f'output/syn_real_eps{int(self.epsilon)}_theta{self.theta}_iter0.dat')
            sampled_data = pd.DataFrame(sampled_data, columns=self.transformed_columns)

            synthetic_data = self.ht.reverse_transform(sampled_data)

            return synthetic_data[self.columns]
