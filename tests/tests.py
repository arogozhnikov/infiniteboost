from __future__ import print_function, division

import numpy
from hep_ml.commonutils import generate_sample
from sklearn.metrics import roc_auc_score
from itertools import combinations_with_replacement

from infiniteboost.researchboosting import ResearchGradientBoostingBase, InfiniteBoosting, InfiniteBoostingWithHoldout
from infiniteboost.researchlosses import MSELoss, LogisticLoss, LambdaLossNDCG, compute_lambdas_numpy, compute_lambdas
from infiniteboost.researchtree import BinTransformer, build_decision_numpy, build_decision
from infiniteboost.fortranfunctions import fortranfunctions


def test_gb_simple():
    X, y = generate_sample(n_samples=10000, n_features=10)
    X = BinTransformer().fit_transform(X)

    reg = ResearchGradientBoostingBase(loss=MSELoss())
    reg.fit(X, y)

    assert roc_auc_score(y, reg.decision_function(X)) > 0.6


def test_reproducibility():
    X, y = generate_sample(n_samples=10000, n_features=10)
    X = BinTransformer().fit_transform(X)
    qids = numpy.sort(numpy.random.randint(0, 100, size=len(X)))

    for loss, target in ([MSELoss(), X.sum(axis=1)],
                         [LogisticLoss(), y],
                         [LambdaLossNDCG(qid_feature=qids, n_threads=2), y * 2]
                         ):
        for model1, model2 in combinations_with_replacement(
                [ResearchGradientBoostingBase, InfiniteBoosting, InfiniteBoostingWithHoldout], r=2):
            reg1 = model1(loss=loss, n_estimators=30, random_state=25).fit(X, target)
            reg2 = model2(loss=loss, n_estimators=30, random_state=25).fit(X, target)
            assert numpy.allclose(reg1.decision_function(X), reg2.decision_function(X)) == (model1 == model2), \
                ('loss', loss)


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
    assert build_decision != build_decision_numpy
    assert build_decision == fortranfunctions.build_decision_fortran


def test_fortran_lambdas(n_documents=500, n_queries=100):
    """ test may fail due to the single/double precision difference """
    qids = numpy.sort(numpy.random.randint(0, n_queries, size=n_documents))
    _, qid_index = numpy.unique(qids, return_index=True)
    orders = numpy.arange(n_documents).astype('int32')

    scores = numpy.random.normal(size=n_documents)
    normalized_gains = numpy.random.normal(size=n_documents)

    for n_threads in [1, 2, 3]:
        print(n_threads)
        grad1, hess1 = compute_lambdas_numpy(
                n_docs=n_documents, qid_indices=qid_index, scores=scores,
                normalized_gains=normalized_gains, n_threads=1, orders=orders)

        grad2, hess2 = fortranfunctions.compute_lambdas_fortran(
            n_docs=n_documents, qid_indices=qid_index, scores=scores,
            normalized_gains=normalized_gains, n_threads=n_threads, orders=orders)

        grad3, hess3 = fortranfunctions.compute_lambdas_fortran(
            n_docs=n_documents, qid_indices=qid_index, scores=scores,
            normalized_gains=normalized_gains, n_threads=1, orders=orders)
        assert numpy.all(grad2 == grad3) and numpy.all(hess2 == hess3)
        assert numpy.allclose(grad1, grad2), [(grad1 - grad2).__abs__().max(), grad1.max()]
        assert numpy.allclose(hess1, hess2)
    assert compute_lambdas != compute_lambdas_numpy
    assert compute_lambdas == fortranfunctions.compute_lambdas_fortran
