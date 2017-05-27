from __future__ import print_function, division, absolute_import

__author__ = 'Alex Rogozhnikov, Tatiana Likhomanenko'

from .researchboosting import ResearchGradientBoostingBase, InfiniteBoosting, InfiniteBoostingWithHoldout
from .researchtree import BinTransformer

from . import researchlosses, researchtree

try:
    from . import fortranfunctions
except:
    raise ImportError('Fortran implementation is not available')

# Substituting functions with their fast implementations
researchlosses.compute_lambdas = fortranfunctions.fortranfunctions.compute_lambdas_fortran
researchtree.build_decision = fortranfunctions.fortranfunctions.build_decision_fortran
