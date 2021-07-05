"""Parent class for all privacy attacks"""

class PrivacyAttack(object):

    def train(self, *args):
        """Train privacy adversary"""
        return NotImplementedError('Method needs to be overwritten by a subclass.')

    def attack(self, *args):
        """Make a guess about target's secret"""
        return NotImplementedError('Method needs to be overwritten by a subclass.')