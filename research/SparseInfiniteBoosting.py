"""
For clear comparison InfiniteBoost and random forest the modification of InfiniteBoost is prepared 
to work with sklearn trees - those can be deep and support sparse data. 
"""
from __future__ import print_function, division, absolute_import

import numpy
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.tree import DecisionTreeRegressor


class InfiniteBoosting(BaseEstimator, ClassifierMixin):
    def __init__(self, capacity=1., n_estimators=10, max_leaf_nodes=None):
        """
        Infinite boosting classifier with fixed capacity value during iterations.
        It supports sparse data (uses sklearn trees).
    
        :param float capacity: capacity of infinite boosting, normalization constant of the ensemble 
        :param int n_estimators: number of estimators in the ensemble 
        :param int max_leaf_nodes: maximal number of leaves in each tree 
        """
        self.n_estimators = n_estimators
        self.capacity = float(capacity)
        self.max_leaf_nodes = max_leaf_nodes

    def encounter_contribution(self, current_sum, contribution, iteration):
        """ Add tree contribution to the ensemble """
        eta = 2. / (iteration + 2)
        return current_sum * (1 - eta) + contribution * eta

    def compute_capacity(self, iteration):
        """ shrinking at early iterations to prevent overstepping """
        eta = 2. / (iteration + 2)
        return min(1 / eta, self.capacity)

    def fit(self, X, y):
        """ train a model """
        assert numpy.all(numpy.in1d(y, [0, 1]))
        y_signed = 2 * y - 1
        n_samples = len(y)
        pred = numpy.zeros(n_samples, dtype='float')
        self.estimators = []

        for iteration in range(self.n_estimators):
            # using AdaLoss -> target = +-1, weights below,
            # result is well-aligned with RF behavior when capacity is zero
            ada_weight = numpy.exp(- y_signed * pred * self.compute_capacity(iteration))

            _indices = numpy.random.RandomState(iteration).randint(0, n_samples, size=n_samples)
            boost_weights = numpy.bincount(_indices, minlength=n_samples) * ada_weight

            tree = DecisionTreeRegressor(max_features='sqrt',
                                         max_depth=None,
                                         random_state=42 * iteration,
                                         max_leaf_nodes=self.max_leaf_nodes)
            tree.fit(X, y_signed, sample_weight=boost_weights)

            pred = self.encounter_contribution(pred, contribution=tree.predict(X), iteration=iteration)
            self.estimators.append(tree)

        return self

    def staged_decision_function(self, X):
        """ Yield predictions after each tree. Raw values, not probabilities """
        pred = numpy.zeros(X.shape[0])
        for iteration, tree in enumerate(self.estimators):
            pred = self.encounter_contribution(
                pred, contribution=tree.predict(X), iteration=iteration)
            yield pred

    def predict_proba(self, X):
        for p in self.staged_decision_function(X):
            pass
        result = numpy.zeros([len(p), 2], dtype=float)
        result[:, 1] = (p + 1.) / 2.
        result[:, 0] = 1 - result[:, 1]
        return result

