import numpy as np


class MinMaxScaler(object):
    def __init__(self):
        self.start_val = None
        self.max_val = None

    def _check_type(self, series):
        pass

    def fit(self, series):
        self._check_type(series)
        self.start_val = series[0]
        self.max_val = np.max(np.abs(series - self.start_val))
        return self

    def transform(self, series):
        return (series - self.start_val) / self.max_val

    def inv_transform(self, series):
        return series * self.max_val + self.start_val

    def fit_transform(self, series):
        self.fit(series)
        return self.transform(series)
