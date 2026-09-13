import statistics

import numpy as np
import scipy.optimize
import scipy.stats


def loss_wasserstein(y_human, y_llm):
    return scipy.stats.wasserstein_distance(y_human, y_llm)


def transform_identity(y_human, y_llm, y_llm_test):
    return y_llm_test


def transform_constant(y_human, y_llm, y_llm_test):
    b = statistics.mean(h - m for h, m in zip(y_human, y_llm))
    return [y + b for y in y_llm_test]


def transform_linear(y_human, y_llm, y_llm_test):
    a = sum(h * m for h, m in zip(y_human, y_llm)) / sum(m * m for m in y_llm)
    return [a * y for y in y_llm_test]


def _ols(y_human, y_llm):
    mean_h, mean_m = statistics.mean(y_human), statistics.mean(y_llm)
    var_m = sum((m - mean_m) ** 2 for m in y_llm)
    if var_m == 0:
        return 0.0, mean_h
    a = sum((m - mean_m) * (h - mean_h) for m, h in zip(y_llm, y_human)) / var_m
    return a, mean_h - a * mean_m


def transform_affine(y_human, y_llm, y_llm_test):
    a, b = _ols(y_human, y_llm)
    return [a * y + b for y in y_llm_test]


def transform_minmax(y_human, y_llm, y_llm_test):
    min_h, max_h = min(y_human), max(y_human)
    min_m, max_m = min(y_llm), max(y_llm)
    if max_m - min_m == 0:
        return y_llm_test
    return [(y - min_m) / (max_m - min_m) * (max_h - min_h) + min_h for y in y_llm_test]


def transform_musigma(y_human, y_llm, y_llm_test):
    mean_h, std_h = statistics.mean(y_human), statistics.stdev(y_human)
    mean_m, std_m = statistics.mean(y_llm), statistics.stdev(y_llm)
    if std_m == 0:
        return y_llm_test
    return [(y - mean_m) / std_m * std_h + mean_h for y in y_llm_test]


def transform_wasserstein_affine(y_human, y_llm, y_llm_test):
    def loss(params):
        a, b = params
        return loss_wasserstein(y_human, [a * y + b for y in y_llm])

    b_init = statistics.mean(h - m for h, m in zip(y_human, y_llm))
    result = scipy.optimize.minimize(loss, [1.0, b_init], method="Nelder-Mead")
    a, b = result.x
    return [a * y + b for y in y_llm_test]


METHODS = [
    ("Identity", transform_identity),
    ("MSE Constant", transform_constant),
    ("MSE Linear", transform_linear),
    ("MSE Affine", transform_affine),
    ("Wass. Affine", transform_wasserstein_affine),
    ("min/max-matching", transform_minmax),
    ("mu/sigma-matching", transform_musigma),
]
