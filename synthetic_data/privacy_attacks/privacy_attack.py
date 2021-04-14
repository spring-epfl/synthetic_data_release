"""Parent class for all privacy attacks"""
from os import path


class PrivacyAttack(object):

    def train(self, *args):
        """Train privacy adversary"""
        return NotImplementedError('Method needs to be overwritten by a subclass.')

    def attack(self, synT):
        """Make a guess about target's secret"""
        return NotImplementedError('Method needs to be overwritten by a subclass.')

    def get_probability_of_success(self, synT, secret):
        """Calculate probability of successfully guessing target's secret"""
        return NotImplementedError('Method needs to be overwritten by a subclass.')
