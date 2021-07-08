"""Parent class for all generative models"""

class GenerativeModel(object):

    def fit(self, data):
        """Fit a generative model to the input dataset"""
        return NotImplementedError('Method needs to be overwritten by a subclass.')

    def generate_samples(self, nsamples):
        """Generate a synthetic dataset of size nsamples"""
        return NotImplementedError('Method needs to be overwritten by a subclass.')