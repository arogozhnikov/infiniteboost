from __future__ import print_function, division, absolute_import
import numpy

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils import check_random_state

__author__ = 'Alex Rogozhnikov, Tatiana Likhomanenko'


def build_decision_numpy(X, targets, weights, bootstrap_weights, current_indices, columns_to_test,
                         depth, n_current_leaves, n_thresh=128, reg=5., use_friedman_mse=True, n_threads=None):
    """ Computes gains for different splits, 
    this function has equivalent to the fortran implementation"""
    n_leaves = 2 ** (depth - 1)
    assert n_current_leaves == n_leaves
    minlength = n_thresh * n_leaves

    leaf_indices = current_indices & (n_leaves - 1)

    hessians = weights * bootstrap_weights
    gradients = targets * hessians

    all_improvements = numpy.zeros([n_leaves, len(columns_to_test), n_thresh])

    for column_index, column in enumerate(columns_to_test):
        indices = leaf_indices.copy()
        indices |= X[:, column].astype(leaf_indices.dtype) << (depth - 1)

        bin_gradients = numpy.bincount(indices, weights=gradients, minlength=minlength).reshape([n_thresh, n_leaves]).T
        bin_hessians = numpy.bincount(indices, weights=weights, minlength=minlength).reshape([n_thresh, n_leaves]).T

        bin_gradients = numpy.cumsum(bin_gradients, axis=1)
        bin_gradients_op = bin_gradients[:, [-1]] - bin_gradients
        bin_hessians = numpy.cumsum(bin_hessians, axis=1)
        bin_hessians_op = bin_hessians[:, [-1]] - bin_hessians

        if not use_friedman_mse:
            # x1 ** 2 / w1 + x2 ** 2 / w2
            improvements = bin_gradients ** 2 / (bin_hessians + reg)
            improvements += bin_gradients_op ** 2 / (bin_hessians_op + reg)
        else:
            # (w1 x2 - x1 w2) ** 2 / w1 / w2 / (w1 + w2)
            improvements = (bin_gradients * bin_hessians_op - bin_gradients_op * bin_hessians) ** 2.
            improvements /= (bin_hessians + reg) * (bin_hessians_op + reg) * (bin_hessians[:, [-1]] + reg)

        all_improvements[:, column_index, :] = improvements
    return all_improvements


build_decision = build_decision_numpy


class ResearchDecisionTree(object):
    _n_thresholds = 128

    def __init__(self, max_depth=3, max_features=1., random_state=42, n_threads=1):
        """
        Quite simple and fast regression decision tree that minimizes MSE.
        Needed when training many trees on large datasets.
        Uses datasets in the binned form both for the sake of speed and 
        minimizing space consumption. 
        To get binned form use BinTransformer (see below). 
        
        Tree always has 2^depth leaves (for computational purposes).
    
        :param int max_depth: maximal depth of tree
        :param float max_features: percentage of feature sampling 
        :param random_state: random state for reproducibility
        :param int n_threads: number of threads 
        """
        self.max_depth = max_depth
        self.max_features = max_features
        self.random_state = random_state
        self.n_threads = n_threads

    def checkX(self, X):
        """Check the input data: size and type"""
        assert isinstance(X, numpy.ndarray), "Type error: {} should be numpy.ndarray".format(type(X))
        assert X.dtype == 'uint8', "Type error: {} is not equal to uint8".format(X.dtype)
        assert numpy.isfortran(X), "Type error: Could not use the data in fortran"

    def fit(self, X, y, sample_weight, check_input=True):
        """
        Train a model
        
        :param X: important! X should be already preprocessed with BinTransform! That is, uint8 and fortran-ordered 
        :param y: target values, 0 and 1 for classification, 
        :param sample_weight: real-valued weights for observations 
        """
        if check_input:
            self.checkX(X)
            assert len(X) == len(y) == len(sample_weight)
        self.random_state = check_random_state(self.random_state)
        _max_features = int(self.max_features * X.shape[1])
        # coding scheme: root = 1, left = 10, right=11, left-right=101, etc.
        current_leaves = numpy.ones(len(X), dtype='int32')
        self.split_features = numpy.zeros(2 ** self.max_depth, dtype='int32')
        self.split_values = numpy.zeros(2 ** self.max_depth, dtype='uint8')

        rows = numpy.arange(len(current_leaves))
        for level in range(self.max_depth):
            selected_features = self.random_state.choice(range(X.shape[1]), size=_max_features, replace=False)
            selected_features = numpy.sort(selected_features)
            n_current_leaves = 2 ** level

            all_improvements = build_decision(
                X, y, weights=sample_weight,
                bootstrap_weights=numpy.ones(len(X), dtype='float32'),
                current_indices=current_leaves,
                columns_to_test=selected_features,
                depth=level + 1,
                n_current_leaves=n_current_leaves,
                n_thresh=self._n_thresholds, reg=0.1, use_friedman_mse=False, n_threads=self.n_threads)

            assert all_improvements.shape == (n_current_leaves, len(selected_features), self._n_thresholds)

            for leaf in numpy.arange(2 ** level):
                leaf_code = 2 ** level + leaf
                feature_id, threshold = numpy.unravel_index(
                    numpy.argmax(all_improvements[leaf]), dims=all_improvements[leaf].shape)
                self.split_features[leaf_code] = selected_features[feature_id]
                self.split_values[leaf_code] = threshold

            is_right = X[rows, self.split_features[current_leaves]] > self.split_values[current_leaves]
            current_leaves = 2 * current_leaves + is_right

        return self

    def transform(self, X):
        """Compute number of leaf in a tree"""
        self.checkX(X)
        current_leaves = numpy.ones(len(X), dtype='int32')
        rows = numpy.arange(len(current_leaves))
        for level in range(self.max_depth):
            is_right = X[rows, self.split_features[current_leaves]] > self.split_values[current_leaves]
            current_leaves = 2 * current_leaves + is_right
        return current_leaves

    def get_n_leaves(self):
        # in fact, first 2 ** max_depth values are never used
        return 2 ** (self.max_depth + 1)


class BinTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, max_bins=128):
        """
        Bin transformer transforms all features (which are expected to be numerical)
        to small integers.
        
        :param int max_bins: maximal number of bins along each axis.
        """
        self.max_bins = max_bins

    def fit(self, X, y=None, sample_weight=None):
        """
        Prepare transformation rule, compute bin edges.
        
        :param X: array-like with data
        :param y: labels, ignored
        :param sample_weight: weights, ignored
        :return: self
        """
        assert self.max_bins < 255, 'Too high number of bins!'
        assert self.max_bins <= ResearchDecisionTree._n_thresholds, 'Too high number of bins for a tree!'
        X = numpy.require(X, dtype='float32')
        self.percentiles_ = []
        for column in range(X.shape[1]):
            values = numpy.array(X[:, column])
            if len(numpy.unique(values)) < self.max_bins:
                self.percentiles_.append(numpy.unique(values)[:-1])
            else:
                targets = numpy.linspace(0, 100, self.max_bins + 1)[1:-1]
                self.percentiles_.append(numpy.percentile(values, targets))
        return self

    def transform(self, X):
        """
        Transform all features to small integers (binarization)
        
        :param X: array-like with data
        :return: numpy.array with transformed features, dtype is 'uint8' for space efficiency.
        """
        X = numpy.require(X, dtype='float32')
        assert X.shape[1] == len(self.percentiles_), 'Wrong names of columns'
        bin_indices = numpy.zeros(X.shape, dtype='uint8', order='F')
        for i, percentiles in enumerate(self.percentiles_):
            bin_indices[:, i] = numpy.searchsorted(percentiles, X[:, i])
        return bin_indices
