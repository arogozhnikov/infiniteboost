from __future__ import print_function, division

import numpy
from hep_ml.commonutils import generate_sample
from sklearn.metrics import roc_auc_score
from itertools import combinations_with_replacement

from infiniteboost.researchboosting import ResearchGradientBoostingBase, InfiniteBoosting, InfiniteBoostingWithHoldout
from infiniteboost.researchlosses import MSELoss, compute_lambdas_numpy
from infiniteboost.researchtree import BinTransformer, build_decision_numpy
from infiniteboost.fortranfunctions import fortranfunctions


def test_gb_simple():
    X, y = generate_sample(n_samples=10000, n_features=10)
    X = BinTransformer().fit_transform(X)

    reg = ResearchGradientBoostingBase(loss=MSELoss())
    reg.fit(X, y)

    assert roc_auc_score(y, reg.decision_function(X)) > 0.6


def test_reprodicibility():
    X, y = generate_sample(n_samples=10000, n_features=10)
    X = BinTransformer().fit_transform(X)

    for model1, model2 in combinations_with_replacement(
            [ResearchGradientBoostingBase, InfiniteBoosting, InfiniteBoostingWithHoldout], r=2):
        reg1 = model1(loss=MSELoss(), random_state=25).fit(X, y)
        reg2 = model2(loss=MSELoss(), random_state=25).fit(X, y)
        assert numpy.allclose(reg1.decision_function(X), reg2.decision_function(X)) == (model1 == model2)


def test_fortran_improvements(n_samples=10000, n_features=10):
    n_samples = numpy.random.poisson(n_samples) + 1
    n_features = numpy.random.poisson(n_features) + 1
    n_thresh = 128
    X = numpy.random.randint(0, n_thresh, size=[n_samples, n_features]).astype(dtype='uint8', order='F')
    targets = numpy.random.normal(size=n_samples).astype('float32')
    weights = numpy.random.uniform(0, 1, size=n_samples)
    bootstrap_weights = numpy.random.poisson(1, size=n_samples).astype('float32')
    depth = numpy.random.randint(1, 7)
    n_current_leaves = 2 ** (depth - 1)
    current_indices = numpy.random.randint(0, n_current_leaves, size=n_samples)
    columns_to_test = numpy.random.choice(n_features, size=(n_features + 1) // 2, replace=False)
    reg = numpy.random.uniform(0, 10)

    for n_threads in [1, 3]:
        for use_friedman in [True, False]:

            improvements1 = build_decision_numpy(
                X, targets, weights, bootstrap_weights, current_indices, columns_to_test,
                depth=depth, n_current_leaves=n_current_leaves,
                n_thresh=n_thresh, reg=reg, use_friedman_mse=use_friedman, n_threads=n_threads)

            improvements2 = fortranfunctions.build_decision_fortran(
                X, targets, weights, bootstrap_weights, current_indices, columns_to_test,
                depth=depth, n_current_leaves=n_current_leaves,
                n_thresh=n_thresh, reg=reg, use_friedman_mse=use_friedman, n_threads=n_threads)

    assert numpy.allclose(improvements1, improvements2)


def test_fortran_lambdas(n_documents=100, n_queries=10):
    """ test may fail due to the single/double precision difference """
    qids = numpy.sort(numpy.random.randint(0, n_queries, size=n_documents))
    _, qid_index = numpy.unique(qids, return_index=True)
    orders = numpy.arange(n_documents).astype('int32')

    scores = numpy.random.normal(size=n_documents)
    normalized_gains = numpy.random.normal(size=n_documents)

    grad1, hess1 = compute_lambdas_numpy(
        n_docs=n_documents, qid_indices=qid_index, scores=scores,
        normalized_gains=normalized_gains, n_threads=1, orders=orders)

    grad2, hess2 = fortranfunctions.compute_lambdas_fortran(
        n_docs=n_documents, qid_indices=qid_index, scores=scores,
        normalized_gains=normalized_gains, n_threads=1, orders=orders)
    assert numpy.allclose(grad1, grad2)
    assert numpy.allclose(hess1, hess2)
