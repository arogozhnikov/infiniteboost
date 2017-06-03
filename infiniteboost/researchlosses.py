"""
This module provides loss definitions used during boosting training.
All losses are prepared as classes with two mandatory functions: 
- `fit` saves necessary information from the training set, for example, sample weights and targets
- `prepare_tree_params` prepares target and sample weights for new tree construction during boosting procedure 
"""
from __future__ import print_function, division, absolute_import
import numpy
from scipy.stats import rankdata
from scipy.special import expit

__author__ = 'Alex Rogozhnikov, Tatiana Likhomanenko'


class MSELoss(object):
    """
    Mean squared loss function 
        \mathbb{L} = \sum_i (y_i - \hat{y_i})^2
    """

    def fit(self, X, y, sample_weight=None):
        """
        Fit the metric 
        
        :param X: data, numpy.array with size [n_samples, n_features] 
        :param y: target, numpy.array with size [n_samples]
        :param sample_weight: sample weights, numpy.array with size [n_samples]
        :return: self 
        """
        self.y = y
        self.sample_weight = sample_weight
        return self

    def prepare_tree_params(self, pred):
        """
        Compute gradients and hessians 

        :param pred: numpy.array with size [n_samples], predictions for each sample 
        :return: negative gradient / hessian, hessian
        """
        return self.y - pred, self.sample_weight


class LogisticLoss(object):
    """ 
    Friedman's version of logistic loss function
        \mathbb{L} = \sum_i \log(1 + e^{-y_i \hat{y_i}})
    """

    def fit(self, X, y, sample_weight=None):
        """
        Fit the metric 

        :param X: data, numpy.array with size [n_samples, n_features] 
        :param y: target, numpy.array with size [n_samples]
        :param sample_weight: sample weights, numpy.array with size [n_samples]
        :return: self 
        """
        assert numpy.all(numpy.in1d(y, [0, 1]))
        self.y_signed = 2 * y - 1
        self.sample_weight = sample_weight
        return self

    def prepare_tree_params(self, pred):
        """
        Compute gradients and hessians 
        
        :param pred: numpy.array with size [n_samples], predictions for each sample 
        :return: negative gradient / hessian, hessian
        """
        tanhs = numpy.tanh(pred)
        ngradients = self.y_signed - tanhs
        hessians = 1 - tanhs ** 2
        return ngradients / hessians, hessians


def optimal_dcg(gains):
    """
    Compute optimal DCG value for gains
    """
    discounts_ = 1 / numpy.log(2 + numpy.arange(len(gains)))
    result = numpy.sum(numpy.sort(gains)[::-1] * discounts_)
    return 1. if result == 0 else result


class LambdaLossNDCG(object):
    """
    Lambda loss and its gradient for nDCG (normalized Discounted Cumulative Gain) measure 
    reproduced according to the paper 
    Burges, C. J. (2010). From ranknet to lambdarank to lambdamart: An overview. Learning, 11(23-581), 81.
    """

    def __init__(self, qid_feature, n_threads):
        self.qid_feature = qid_feature
        self.n_threads = n_threads

    def fit(self, X, y, sample_weight=None):
        """
        Fit the metric 

        :param X: data, numpy.array with size [n_samples, n_features] 
        :param y: target, numpy.array with size [n_samples]
        :param sample_weight: sample weights, numpy.array with size [n_samples]
        :return: self 
        """

        if isinstance(self.qid_feature, str):
            qids = numpy.asarray(X[self.qid_feature])
        else:
            qids = numpy.array(self.qid_feature)
            assert len(y) == len(qids)
        # check that query id does not decrease in the dataset
        assert numpy.all(numpy.diff(qids) >= 0)
        _, self.qid_index = numpy.unique(qids, return_index=True)
        self.n_qids = len(self.qid_index)
        # add the last index for later usage [qid_index, qid_index + 1]
        # to get the access to the documents with the same query id
        self.qid_index = numpy.insert(self.qid_index, len(self.qid_index),
                                      len(qids)).astype('int32')
        # compute (2^relevance - 1) / IDCG(query) for each document
        # (to use it later for NDCG computing)
        self.normalized_gains = numpy.array(2 ** y - 1., dtype='float32')

        for qid in range(self.n_qids):
            left, right = self.qid_index[qid], self.qid_index[qid + 1]
            # check that all documents have the same query id
            assert numpy.allclose(numpy.diff(qids[left:right]), 0)
            self.normalized_gains[left:right] /= optimal_dcg(self.normalized_gains[left:right])

        self.sample_weight = numpy.ones(len(y))
        self.orders = numpy.arange(len(y)).astype('int32')
        return self

    def prepare_tree_params(self, pred):
        """
        Compute lambda gradients and hessians 

        :param pred: numpy.array with size [n_samples], predictions for each sample 
        :return: negative gradient / hessian, hessian
        """
        gradients, hessians = compute_lambdas(
            n_docs=len(pred),
            qid_indices=self.qid_index,
            scores=pred,
            normalized_gains=self.normalized_gains,
            n_threads=self.n_threads,
            orders=self.orders
        )

        hessians += 1e-4
        return gradients / hessians, hessians


def compute_lambdas_numpy(n_docs, qid_indices, scores,
                          normalized_gains, n_threads, orders):
    """
    Implementation of lambda gradients with numpy.
    Details in Donmez, P., Svore, K. M., & Burges, C. J. (2009, July). On the local optimality of LambdaRank. In Proceedings of the 32nd international ACM SIGIR conference on Research and development in information retrieval (pp. 460-467). ACM.
    """
    assert len(scores) == len(normalized_gains) == n_docs
    gradients = numpy.zeros(n_docs)
    hessians = numpy.zeros(n_docs)

    for qid in range(len(qid_indices) - 1):
        left, right = qid_indices[qid], qid_indices[qid + 1]
        gains = normalized_gains[left:right]
        gains_diff = numpy.subtract.outer(gains, gains)
        s_ij = numpy.sign(gains_diff)

        preds = scores[left:right]
        ranks = rankdata(-preds, method='ordinal')
        discounts = 1. / numpy.log2(1 + ranks)
        discount_diff = numpy.subtract.outer(discounts, discounts)
        prediction_diff = numpy.subtract.outer(preds, preds)
        sigmoids = expit(- s_ij * prediction_diff)
        lambdas = s_ij * numpy.abs(gains_diff * discount_diff * sigmoids)

        gradients[left:right] = lambdas.sum(axis=1)
        # that's how lightGBM and XGBoost approximate hessian
        hessians[left:right] = numpy.abs(lambdas * (1. - sigmoids)).sum(axis=1)

    return gradients, hessians


compute_lambdas = compute_lambdas_numpy


class NDCG_metric(object):
    """
    nDCG (normalized Discounted Cumulative Gain) measure for ranking
    """
    def __init__(self, queries, relevances, maximal=100000):
        self.queries = queries
        self.relevances = relevances
        assert numpy.all(numpy.diff(queries) >= 0)
        _, self.qid_index, self.queries = numpy.unique(
            self.queries, return_index=True, return_inverse=True)
        prev_q = None
        ranks = numpy.zeros(len(queries), dtype='int')
        # define available ranks for each query
        for i, q in enumerate(self.queries):
            if q != prev_q:
                ranks[i] = 0
                prev_q = q
            else:
                ranks[i] = ranks[i - 1] + 1
        gains = 2 ** relevances - 1
        # normalizing for empty query
        query_gains = numpy.bincount(self.queries, weights=gains)
        gains[query_gains[self.queries] == 0] = 1.
        # compute available discounts
        discounts = 1. / numpy.log2(2 + ranks) * (ranks < maximal)
        # sort documents by relevance separately in each query
        ideal_orders = numpy.argsort(self.queries * 100 - relevances)
        assert numpy.allclose(self.queries, self.queries[ideal_orders])
        dcg_ideal_gains = discounts * gains[ideal_orders]
        # ideal DCG for query
        query_gains = numpy.bincount(self.queries, weights=dcg_ideal_gains)
        query_gains = numpy.maximum(query_gains, 1e-6)
        self.normalized_discounts = discounts
        self.gains = gains / query_gains[self.queries]

    def compute(self, scores):
        # sort documents by predicted relevance separately in each query
        # lazy tricky way to avoid sorting on two columns
        coeff = 0.4 / numpy.max(numpy.abs(scores))
        orders = numpy.argsort(self.queries - scores * coeff)
        assert numpy.allclose(self.queries, self.queries[orders])
        return numpy.sum(self.gains[orders] * self.normalized_discounts) / len(set(self.queries))
