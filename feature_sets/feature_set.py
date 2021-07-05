"""A parent class for all feature extraction layers"""


class FeatureSet(object):
    def extract(self, data):
        return NotImplementedError('Method needs to be overwritten by subclass')