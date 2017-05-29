from __future__ import print_function, division, absolute_import

import copy

import numpy
from sklearn.base import BaseEstimator
from sklearn.utils import check_random_state

from .researchlosses import LambdaLossNDCG
from .researchtree import ResearchDecisionTree

__author__ = 'Alex Rogozhnikov, Tatiana Likhomanenko'


class ResearchGradientBoostingBase(BaseEstimator):
    """
    Base class for gradient boosting estimators. 
    Implements canonical gradient boosting. 
    """

    def __init__(self, loss=None,
                 n_estimators=100,
                 learning_rate=0.1,
                 subsample=1.,
                 max_features=1.,
                 max_depth=3,
                 n_threads=2,
                 l2_regularization=5.,
                 use_all_in_update=False,
                 random_state=42):
        """
        `max_depth`, `max_features` are parameters of regression tree, which are used as base estimator.

        :param bool use_all_in_update: if true, all the data are used in setting new leaves values
        :param loss: any descendant of AbstractLossFunction, those are very various.
            See :class:`hep_ml.losses` for available losses.
        :type loss: AbstractLossFunction
        :param int n_estimators: number of trained trees.
        :param float subsample: fraction of data to use on each stage, or "bagging" for bagging strategy
        :param float learning_rate: size of step.
        """
        self.loss = loss
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.subsample = subsample
        self.max_features = max_features
        self.max_depth = max_depth
        self.n_threads = n_threads
        self.l2_regularization = l2_regularization
        self.use_all_in_update = use_all_in_update
        self.random_state = random_state

    def _check_params(self):
        """Checking parameters of classifier set in __init__"""
        assert self.loss is not None, 'loss was not set'
        assert self.n_estimators > 0, 'n_estimators should be positive'
        assert 0 < self.subsample <= 1., 'subsample should be in (0, 1]'
        self.random_state = check_random_state(self.random_state)

    def _estimate_tree(self, tree, leaf_values, X):
        """taking indices of leaves and return the corresponding value for each sample"""
        leaves = tree.transform(X)
        return leaf_values[leaves]

    def generate_weights(self, n_samples):
        """generate weights for samples: bagging (with replacement) or subsampling (without replacement)"""
        if self.subsample == 'bagging':
            return numpy.bincount(self.random_state.randint(0, n_samples, size=n_samples), minlength=n_samples)
        else:
            n_inbag = int(self.subsample * n_samples)
            train_indices = self.random_state.choice(n_samples, size=n_inbag, replace=False)
            weights = numpy.zeros(n_samples)
            weights[train_indices] = 1
            return weights

    def encounter_contribution(self, current_sum, contribution, iteration, n_iterations,
                               is_training=None, bootstrap_weights=None):
        """function to be modified to experiment with different algorithms """
        return current_sum + self.learning_rate * contribution

    def compute_initial_step(self, n_samples):
        """compute initial approximation"""
        initial_step = 0
        for _ in range(10):
            pred = numpy.zeros(n_samples) + initial_step
            target, weight = self.loss.prepare_tree_params(pred)
            initial_step += numpy.average(target, weights=weight)
        return initial_step

    def fit(self, X, y, sample_weight=None):
        """
        Train a model
        :param X: important! X should be already after BinTransform! That is, uint8 and fortran-ordered 
        :param y: target values, 0 and 1 for classification, 
        :param sample_weight: real-valued weights for observations 
        """
        self._check_params()
        n_samples = len(X)

        # checking arguments for loss function
        if sample_weight is None:
            sample_weight = numpy.ones(len(y), dtype='float')
        assert len(X) == len(y) == len(sample_weight)

        self.loss = copy.deepcopy(self.loss)
        self.loss.fit(X, y, sample_weight=sample_weight)

        self.estimators = []
        self.n_features = X.shape[1]

        self.initial_step = self.compute_initial_step(n_samples)
        y_pred = numpy.zeros(n_samples, dtype=float) + self.initial_step

        for iteration in range(self.n_estimators):
            tree = ResearchDecisionTree(
                max_depth=self.max_depth,
                max_features=self.max_features,
                random_state=self.random_state,
                n_threads=self.n_threads,
            )

            # tree learning
            residual, weights = self.loss.prepare_tree_params(y_pred)

            bootstrap_weights = self.generate_weights(n_samples=n_samples)
            bootstrapped_weights = weights * bootstrap_weights
            tree.fit(X, residual, sample_weight=bootstrapped_weights, check_input=False)

            # compute values in tree leaves
            leaf_values = self.prepare_new_leaves_values(
                tree, X=X, residual=residual,
                weights=weights if self.use_all_in_update else bootstrapped_weights)

            # encounter contribution
            y_pred = self.encounter_contribution(
                y_pred,
                contribution=self._estimate_tree(tree, leaf_values=leaf_values, X=X),
                iteration=iteration, n_iterations=self.n_estimators,
                is_training=True, bootstrap_weights=bootstrap_weights)

            self.estimators.append([tree, leaf_values])
        return self

    def prepare_new_leaves_values(self, tree, X, residual, weights):
        """define leaf value as \sum grad_i / (\sum hessian_i + regularization)"""
        terminal_regions = tree.transform(X)
        minlength = tree.get_n_leaves()
        nominators = numpy.bincount(terminal_regions, weights=residual * weights, minlength=minlength)
        return nominators / (
            numpy.bincount(terminal_regions, weights=weights, minlength=minlength) + self.l2_regularization)

    def staged_decision_function(self, X):
        """Raw output, sum of trees' predictions after each iteration.

        :param X: data
        :return: sequence of numpy.array of shape [n_samples]
        """
        y_pred = numpy.zeros(len(X)) + self.initial_step
        for iteration, (tree, leaf_values) in enumerate(self.estimators):
            y_pred = self.encounter_contribution(
                y_pred, contribution=self._estimate_tree(tree, leaf_values=leaf_values, X=X),
                iteration=iteration, n_iterations=self.n_estimators,
                is_training=False, bootstrap_weights=None
            )
            yield y_pred

    def decision_function(self, X):
        """Raw output, sum of trees' predictions

        :param X: data
        :return: numpy.array of shape [n_samples]
        """
        result = None
        for score in self.staged_decision_function(X):
            result = score
        return result


class InfiniteBoosting(ResearchGradientBoostingBase):
    def __init__(self, capacity=100., **kargs):
        self.capacity = capacity
        if 'learning_rate' in kargs:
            print('warning: learning rate is ignored in forest regressor')
        ResearchGradientBoostingBase.__init__(self, **kargs)

    def encounter_contribution(self, current_sum, contribution, iteration, n_iterations,
                               is_training=None, bootstrap_weights=None):
        """
        Adding new tree to the ensemble using rule:
        F(x) \gets capacity \times\varepsilon_m\times\text{tree}_m(x) + (1-\varepsilon_m)F(x), where
        \varepsilon_m = \frac{\alpha_m}{\sum_{k=1}^m \alpha_k}, alpha_k = k
        """
        epsilon = 2. / (iteration + 2)
        result = current_sum * (1 - epsilon) + min(epsilon * self.capacity, 1) * contribution
        # compensating decay of initial step
        result += epsilon * self.initial_step
        return result


class InfiniteBoostingWithHoldout(ResearchGradientBoostingBase):
    """Using 5% holdout to estimate capacity in infinite regressor"""
    _holdout_fraction = 0.05

    def generate_weights(self, n_samples):
        if not hasattr(self, 'is_not_holdout'):
            if not isinstance(self.loss, LambdaLossNDCG):
                self.is_not_holdout = numpy.random.RandomState(1337).uniform(size=n_samples) > self._holdout_fraction
                self.is_holdout = 1 - self.is_not_holdout
            else:
                selected_qids = numpy.random.RandomState(1337).choice(
                    self.loss.n_qids, size=int(self._holdout_fraction * self.loss.n_qids))
                print(len(selected_qids), ' queries in holdout')
                self.is_holdout = numpy.in1d(self.loss.qid_feature, selected_qids)
                self.is_not_holdout = 1 - self.is_holdout

        weights = ResearchGradientBoostingBase.generate_weights(self, n_samples=n_samples)
        return self.is_not_holdout * weights

    def encounter_contribution(self, current_sum, contribution, iteration, n_iterations,
                               is_training=None, bootstrap_weights=None):
        """ Function handles both encountering contributions and correcting capacity  """
        if is_training and (iteration == 0):
            self.capacities = [0.5]
        if iteration == 0:
            self.all_contributions = numpy.zeros(len(contribution), dtype='float64')

        # \sum_i (i * tree_i) / (\sum_i i) * capacity
        next_normalisation = (iteration + 2) * (iteration + 1) / 2.
        self.all_contributions += (iteration + 1) * contribution
        current_predictions = self.initial_step + \
                              self.all_contributions * (self.capacities[iteration] / next_normalisation)

        if is_training:
            # correcting capacity for the next step
            target, weights = self.loss.prepare_tree_params(current_predictions)
            total_sign = numpy.sign(numpy.sum(target * weights * current_predictions * self.is_holdout))

            new_capacity = self.capacities[-1] * ((iteration + 2) / (iteration + 1.)) ** total_sign
            self.capacities.append(new_capacity)

        return current_predictions
