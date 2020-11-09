# Copyright 2020 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Statistical tests."""

from bisect import bisect_left
from collections import namedtuple
from typing import List
import sys

import numpy as np
import pandas as pd
import scikit_posthocs as sp
import scipy.stats as ss
from scipy.special import betaln

SIGNIFICANCE_THRESHOLD = 0.05
A12_EFFECT_THRESHOLD = 0.5735


def _create_p_value_table(benchmark_snapshot_df,
                          statistical_test,
                          alternative="two-sided",
                          statistic='pvalue'):
    """Given a benchmark snapshot data frame and a statistical test function,
    returns a p-value table. The |alternative| parameter defines the alternative
    hypothesis to be tested. Use "two-sided" for two-tailed (default), and
    "greater" or "less" for one-tailed test.

    The p-value table is a square matrix where each row and column represents a
    fuzzer, and each cell contains the resulting p-value of the pairwise
    statistical test of the fuzzer in the row and column of the cell.
    """

    def test_pair(measurements_x, measurements_y):
        return statistical_test(measurements_x,
                                measurements_y,
                                alternative=alternative)

    groups = benchmark_snapshot_df.groupby('fuzzer')
    samples = groups['edges_covered'].apply(list)
    fuzzers = samples.index

    data = []
    for f_i in fuzzers:
        row = []
        for f_j in fuzzers:
            value = np.nan
            if f_i != f_j and set(samples[f_i]) != set(samples[f_j]):
                res = test_pair(samples[f_i], samples[f_j])
                value = getattr(res, statistic, np.nan)
            row.append(value)
        data.append(row)

    return pd.DataFrame(data, index=fuzzers, columns=fuzzers)


def exp_pair_test(experiment_snapshot_df, benchmark, f1, f2):
    df = experiment_snapshot_df[experiment_snapshot_df.benchmark == benchmark]
    x = df[df.fuzzer == f1].edges_covered
    y = df[df.fuzzer == f2].edges_covered
    if len(x) < 1 or len(y) < 1:
        print(f"[-] (pair_test) NOT enough samples for {benchmark},{f1},{f2} ")
        return Bunch(pvalue=1, a12=0, statistic=0)
    return r_mannwhitneyu(x, y)


def one_sided_u_test(benchmark_snapshot_df):
    """Returns p-value table for one-tailed Mann-Whitney U test."""
    return _create_p_value_table(benchmark_snapshot_df,
                                 ss.mannwhitneyu,
                                 alternative='greater')


def two_sided_u_test(benchmark_snapshot_df):
    """Returns p-value table for two-tailed Mann-Whitney U test."""
    return _create_p_value_table(benchmark_snapshot_df,
                                 ss.mannwhitneyu,
                                 alternative='two-sided')


def two_sided_u_test_exact(benchmark_snapshot_df):
    """Returns p-value table for two-tailed Mann-Whitney U test."""
    return _create_p_value_table(benchmark_snapshot_df,
                                 mwu, alternative='two-sided')


def two_sided_u_test_r(benchmark_snapshot_df):
    """Returns p-value table for two-tailed Mann-Whitney U test."""
    return _create_p_value_table(benchmark_snapshot_df,
                                 r_mannwhitneyu,
                                 alternative='two-sided')


def vda_measure(benchmark_snapshot_df):
    """Returns A12 measure table for Vargha-Delaney A12."""
    return _create_p_value_table(benchmark_snapshot_df,
                                 r_mannwhitneyu,
                                 alternative='two-sided',
                                 statistic='a12')


def one_sided_wilcoxon_test(benchmark_snapshot_df):
    """Returns p-value table for one-tailed Wilcoxon signed-rank test."""
    return _create_p_value_table(benchmark_snapshot_df,
                                 ss.wilcoxon,
                                 alternative='greater')


def two_sided_wilcoxon_test(benchmark_snapshot_df):
    """Returns p-value table for two-tailed Wilcoxon signed-rank test."""
    return _create_p_value_table(benchmark_snapshot_df,
                                 ss.wilcoxon,
                                 alternative='two-sided')


def anova_test(benchmark_snapshot_df):
    """Returns p-value for ANOVA test.

    Results should only considered when we can assume normal distributions.
    """
    groups = benchmark_snapshot_df.groupby('fuzzer')
    sample_groups = groups['edges_covered'].apply(list).values

    _, p_value = ss.f_oneway(*sample_groups)
    return p_value


def anova_posthoc_tests(benchmark_snapshot_df):
    """Returns p-value tables for various ANOVA posthoc tests.

    Results should considered only if ANOVA test rejects null hypothesis.
    """
    common_args = {
        'a': benchmark_snapshot_df,
        'group_col': 'fuzzer',
        'val_col': 'edges_covered',
        'sort': True
    }
    p_adjust = 'holm'

    posthoc_tests = {}
    posthoc_tests['student'] = sp.posthoc_ttest(**common_args,
                                                equal_var=False,
                                                p_adjust=p_adjust)
    posthoc_tests['turkey'] = sp.posthoc_tukey(**common_args)
    return posthoc_tests


def kruskal_test(benchmark_snapshot_df):
    """Returns p-value for Kruskal test."""
    groups = benchmark_snapshot_df.groupby('fuzzer')
    sample_groups = groups['edges_covered'].apply(list).values

    _, p_value = ss.kruskal(*sample_groups)
    return p_value


def kruskal_posthoc_tests(benchmark_snapshot_df):
    """Returns p-value tables for various Kruskal posthoc tests.

    Results should considered only if Kruskal test rejects null hypothesis.
    """
    common_args = {
        'a': benchmark_snapshot_df,
        'group_col': 'fuzzer',
        'val_col': 'edges_covered',
        'sort': True
    }
    p_adjust = 'holm'

    posthoc_tests = {}
    posthoc_tests['mann_whitney'] = sp.posthoc_mannwhitney(**common_args,
                                                           p_adjust=p_adjust)
    posthoc_tests['conover'] = sp.posthoc_conover(**common_args,
                                                  p_adjust=p_adjust)
    posthoc_tests['wilcoxon'] = sp.posthoc_wilcoxon(**common_args,
                                                    p_adjust=p_adjust)
    posthoc_tests['dunn'] = sp.posthoc_dunn(**common_args, p_adjust=p_adjust)
    posthoc_tests['nemenyi'] = sp.posthoc_nemenyi(**common_args)

    return posthoc_tests


def friedman_test(experiment_pivot_df):
    """Returns p-value for Friedman test."""
    pivot_df_as_matrix = experiment_pivot_df.values
    _, p_value = ss.friedmanchisquare(*pivot_df_as_matrix)
    return p_value


def friedman_posthoc_tests(experiment_pivot_df):
    """Returns p-value tables for various Friedman posthoc tests.

    Results should considered only if Friedman test rejects null hypothesis.
    """
    posthoc_tests = {}
    posthoc_tests['conover'] = sp.posthoc_conover_friedman(experiment_pivot_df)
    posthoc_tests['nemenyi'] = sp.posthoc_nemenyi_friedman(experiment_pivot_df)
    return posthoc_tests


MannwhitneyuResult = namedtuple('MannwhitneyuResult', ('statistic', 'pvalue'))


class Bunch:
    def __init__(self, **kwds):
        self.__dict__.update(kwds)


# TODO: add paired=False, do a signed-rank test if paired=True or y is
#       not provided. See morestats.wilcoxon
def mwu(x, y, correction=True, exact='auto', alternative='two-sided'):
    """
    Computes two-sample unpaired Mann-Whitney-Wilcoxon tests.

    Parameters
    ----------
    x : array_like, 1-D
        The first set of measurements.
    y : array_like, 1-D
        The second set of measurements.
    correction : bool, optional
        If True, apply continuity correction by adjusting the Wilcoxon rank
        statistic by 0.5 towards the mean value when computing the z-statistic.
        Default is True.
    exact : {True, False, 'auto'}, optional
        Whether an exact test should be performed on the data. See notes for
        default behavior.
    alternative : str, optional
        Whether a two-tailed or one-tailed test should be performed on the
        supplied vectors. Arguments are 'two-tailed', 'less', and 'greater'.
        Default is 'two-tailed'.

    Returns
    -------
    Bunch object with following attributes:
        statistic : float
            The test statistic.
        pvalue : float
            The pvalue of the test.
        exact : bool
            Indicates if an exact pvalue was calculated.
        alternative : str
            Describes the alternative hypothesis.
        u1 : float
            The U-value corresponding to the set of measurements in x
        u2 : float
            The U-value corresponding to the set of measurements in y

    Notes
    -----
    Exact tests should be used for smaller sample sizes. Concretely, as
    len(x) and len(y) increase to and beyond 8, the distribution of U differs
    negligibly from the normal distribution[1]_. The default behavior of this
    test is to calculate the number of possible sequences for inputs of
    length(x) and length(y), and to do an exact calculation if the number of
    possible combinations is <100000. The default behavior may be overridden
    with the use_exact flag.

    If an exact test is not performed, the U-distribution is approximated as a
    normal distribution.

    This test corrects for ties and by default uses a continuity correction
    when approximating the U statistic distribution.

    The reported p-value is for a two-sided hypothesis. To get the one-sided
    p-value set alternative to 'greater' or 'less' (default is 'two-sided').

    For Mann-Whitney-Wilcoxon tests, the reported U statistic is the U used to
    test the hypothesis. The u1 and u2 statistics are returned as well,
    corresponding to x and y, respectively.

    .. versionadded:: 0.17.0

    References
    ----------
    .. [1] H.B. Mann and D.R. Whitney, "On a test of whether one of two random
           variables is stochastically larger than the other", The Annals of
           Mathematical Statistics, Vol. 18, pp. 50-60, 1947.
           DOI:10.1214/aoms/1177730491
    .. [2] http://en.wikipedia.org/wiki/Mann-Whitney_U_test

    """
    if alternative not in ('two-sided', 'less', 'greater'):
        raise AttributeError("Alternative should be one of: "
                             "'two-sided', 'less', or 'greater'")

    x = np.asarray(x)
    y = np.asarray(y)
    if x.ndim > 1 or x.ndim > 1:
        raise ValueError('Expected 1-d arrays.')

    n1, n2 = x.size, y.size
    if exact == 'auto':
        maxn = 100000
        exact = ((n1 < 21 or n2 < 21) and n1 + n2 < maxn)
        exact = exact and (-np.log(n1 + n2 + 1) -
                            betaln(n1 + 1, n2 + 1) < np.log(maxn))

    ranked = ss.rankdata(np.concatenate((x, y)))
    rankx = ranked[0:n1]       # get the x-ranks
    T = ss.tiecorrect(ranked)
    if T == 0:
        raise ValueError('All numbers are identical')

    # Compute the A12 measure
    R1 = rankx.sum()
    # A = (R1/n1 - (n1+1)/2)/n2 # formula (14) in Vargha and Delaney, 2000
    A = (2 * R1 - n1 * (n1 + 1)) / (2 * n2 * n1)  # equivalent formula to avoid accuracy errors
    # Cliff's delta linearly related to V-D A12 measure (range (-1, 1))
    CD = (A * 2) -1

    # Given ranks, calculate U for x; remainder is U for y
    u1 = rankx.sum() - n1*(n1+1)/2
    u2 = n1*n2 - u1
    if not exact:
        u1, u2 = u2, u1   # 'exact' code path calculates an empirical cdf
                          # normal approx uses sf instead

    if alternative == 'two-sided':
        bigu, smallu = max(u1, u2), min(u1, u2)
    elif alternative == 'greater':
        bigu, smallu = u2, u1
    else:
        bigu, smallu = u1, u2

    # For small sample sizes, do an exact calculation; otherwise use the
    # normal approximation.
    if exact:
        a = np.arange(n1, n1 + n2)
        a_range = np.arange(n2)
        u = [0]
        while a.sum() != n2*(n2-1)/2:   # When in leftmost position, a == list(range(n2))
            # Do the shift operation
            i = np.nonzero(a - a_range)[0][0]
            a[:i+1] = a[i] + np.arange(-i-1, 0)

            # count(a < a1) = U2
            u1 = n1*n2 + n2*(n2-1)/2 - a.sum()
            u2 = n1*n2 - u1
            # store min U value to array
            val = min(u1, u2) if alternative == 'two-sided' else u1
            u.append(val)
        u = np.array(u)
        p = np.count_nonzero(u <= smallu) / u.size
    else:
        sd = np.sqrt(T * n1 * n2 * (n1 + n2 + 1) / 12.0)
        c = -0.5 if correction else 0
        z = (bigu + c - n1*n2/2.0) / sd
        if alternative == 'two-sided':
            p = 2 * ss.norm.sf(abs(z))
        else:
            p = ss.norm.sf(z)

    dct = {'two-sided': 'x and y are sampled from different populations',
           'less': 'x is sampled from a population of smaller values than y',
           'greater': 'x is sampled from a population of larger values than y'}
    return Bunch(statistic=smallu,
                 pvalue=p,
                 alternative=dct[alternative],
                 a12=A, CD=CD,
                 u1=u1, u2=u2)


# Vargha-Delaney A12
# from https://gist.github.com/jacksonpradolima/f9b19d65b7f16603c837024d5f8c8a65
def VD_A(treatment: List[float], control: List[float]):
    """
    Computes Vargha and Delaney A index
    A. Vargha and H. D. Delaney.
    A critique and improvement of the CL common language
    effect size statistics of McGraw and Wong.
    Journal of Educational and Behavioral Statistics, 25(2):101-132, 2000
    The formula to compute A has been transformed to minimize accuracy errors
    See: http://mtorchiano.wordpress.com/2014/05/19/effect-size-of-r-precision/
    :param treatment: a numeric list
    :param control: another numeric list
    :returns the value estimate and the magnitude
    """
    m = len(treatment)
    n = len(control)

#   if m != n:
#       raise ValueError("Data d and f must have the same length")

    r = ss.rankdata(treatment + control)
    r1 = sum(r[0:m])

    # Compute the measure
    # A = (r1/m - (m+1)/2)/n # formula (14) in Vargha and Delaney, 2000
    A = (2 * r1 - m * (m + 1)) / (2 * n * m)  # equivalent formula to avoid accuracy errors

    levels = [0.147, 0.33, 0.474]  # effect sizes from Hess and Kromrey, 2004
    magnitude = ["negligible", "small", "medium", "large"]
    scaled_A = (A - 0.5) * 2

    magnitude = magnitude[bisect_left(levels, abs(scaled_A))]
    estimate = A

    return estimate, magnitude


import rpy2.robjects as robjects
_mann_whitneyu_r = robjects.r['wilcox.test']

robjects.r("""
AMeasure <- function(a,b){
    # Compute the rank sum (Eqn 13)
    r = rank(c(a,b))
    r1 = sum(r[seq_along(a)])

    # Compute the measure (Eqn 14)
    m = length(a)
    n = length(b)
    #  A = (r1/m - (m+1)/2)/n
    A = (2* r1 - m*(m+1))/(2*m*n)

    A
}""")
vd_a = robjects.r['AMeasure']

def r_mannwhitneyu(x, y, exact=True, alternative="two.sided"):
    if len(x) < 1 or len(y) < 1:
        return Bunch(pvalue=1, a12=0, statistic=0)
    if alternative == 'two-sided':
        alternative = 'two.sided'
    v1 = robjects.IntVector(x)
    v2 = robjects.IntVector(y)
    try:
        wres = _mann_whitneyu_r(v1, v2, exact=exact, alternative=alternative)
    except:
        print(x,y)
        sys.exit(1)
    A = vd_a(v1, v2)[0]
    uval = wres[0][0]
    pval = wres[2][0]
    return Bunch(pvalue=pval, a12=A, statistic=uval, u1=uval)
