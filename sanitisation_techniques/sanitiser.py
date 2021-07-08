""" Parent class for sanitisers """
from pandas import DataFrame, cut
from sklearn.impute import SimpleImputer
from pandas.api.types import is_numeric_dtype

from utils.constants import *

class Sanitiser(object):

    def sanitise(self, data):
        """ Apply a privacy policy to the data. """
        return NotImplementedError('Method needs to be overwritten by a subclass')


class SanitiserNHS(Sanitiser):
    """ A sanitisation mechanism that follows the strategy described by NHS England. """
    def __init__(self, metadata,
                 nbins=10, thresh_rare=0,
                 max_quantile = 1, anonymity_set_size=1,
                 drop_cols=None, quids=None):

        self.metadata = self._read_meta(metadata, drop_cols, quids)
        self.datatype = DataFrame

        self.histogram_size = nbins
        self.unique_threshold = thresh_rare
        self.quids = quids
        self.max_quantile = max_quantile
        self.anonymity_set_size = anonymity_set_size

        self.ImputerCat = SimpleImputer(strategy='most_frequent')
        self.ImputerNum = SimpleImputer(strategy='median')

        self.trained = False

        self.__name__ = f'SanitiserNHSk{self.anonymity_set_size}'

    def sanitise(self, data):
        """
        Sanitise a sensitive dataset

        :param data: DataFrame: Sensitive raw dataset
        :return: san_data: DataFrame: Sanitised dataset
        """
        san_data = DataFrame(index=data.index)
        data = self._impute_missing_values(data)
        drop_records = []

        for col, cdict in self.metadata.items():
            coltype = cdict['type']
            col_data = data[col].copy()

            if coltype == FLOAT or coltype == INTEGER:
                col_data = col_data.astype(int)

                # Cap numerical attributes
                cap = col_data.quantile(self.max_quantile)
                idx = col_data[col_data > cap].index
                col_data.loc[idx] = int(cap)

            elif coltype == CATEGORICAL or coltype == ORDINAL:
                if is_numeric_dtype(col_data):
                    # Bins numerical cols marked as quid into specified bins
                    col_data = cut(col_data, bins=cdict['bins'], labels=cdict['categories'])
                    col_data = col_data.astype(str)

                # Remove any records with rare categories
                frequencies = col_data.value_counts()
                drop_cats = frequencies[frequencies <= self.unique_threshold].index

                for c in drop_cats:
                    ridx = list(col_data[col_data == c].index)
                    drop_records.extend(ridx)

            san_data[col] = col_data.values

        drop_records = list(set(drop_records))
        san_data = san_data.drop(drop_records)

        # Enforce k-anonymity constraint
        if self.quids is not None:
            anonymity_sets = san_data.groupby(self.quids).size()
            groups = anonymity_sets[anonymity_sets < self.anonymity_set_size].index
            for g in groups:
                conditions = [f"{k} == '{v}'" for k,v in zip(self.quids, g)]
                query = " and ".join(conditions)
                didx = san_data.query(query).index
                san_data = san_data.drop(didx)

        return san_data

    def _read_meta(self, metadata, drop_cols, quids):
        """ Read metadata from metadata file."""
        if quids is None:
            quids = []

        if drop_cols is None:
            drop_cols = []

        metadict = {}

        for cdict in metadata['columns']:
            col = cdict['name']
            coltype = cdict['type']

            if col not in drop_cols:
                if coltype == FLOAT or coltype == INTEGER:
                    if col in quids:
                        cbins = cdict['bins']
                        cats = [f'({cbins[i]},{cbins[i+1]}]' for i in range(len(cbins)-1)]

                        metadict[col] = {
                            'type': CATEGORICAL,
                            'categories': cats,
                            'bins': cbins,
                            'size': len(cats)
                        }

                    else:
                        metadict[col] = {
                            'type': coltype,
                            'min': cdict['min'],
                            'max': cdict['max']
                        }

                elif coltype == CATEGORICAL or coltype == ORDINAL:
                    metadict[col] = {
                        'type': coltype,
                        'categories': cdict['i2s'],
                        'size': len(cdict['i2s'])
                    }

                else:
                    raise ValueError(f'Unknown data type {coltype} for attribute {col}')

        return metadict

    def _impute_missing_values(self, df):
        df_impute = df.copy()

        cat_cols = []
        num_cols = []

        for col, cdict in self.metadata.items():
            if col in list(df_impute):
                if cdict['type'] in [CATEGORICAL, ORDINAL]:
                    cat_cols.append(col)

                elif cdict['type'] in NUMERICAL:
                    num_cols.append(col)

        self.ImputerCat.fit(df[cat_cols])
        df_impute[cat_cols] = self.ImputerCat.transform(df[cat_cols])

        self.ImputerNum.fit(df[num_cols])
        df_impute[num_cols] = self.ImputerNum.transform(df[num_cols])

        return df_impute