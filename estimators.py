"""Estimators for the ATE and odds ratios (OR)."""

import numpy as np
import math
from scipy.special import expit
from sklearn.linear_model import LogisticRegression

def get_stratum_indices(X_samples, X_stratum_value):
    """Returns the indices of X_samples where the sample x == X_stratum_value."""
    return np.where(np.prod(X_samples == X_stratum_value, axis=1))

def get_unstratified_OR_estimate(Z_samples, X_samples, Y_samples):
    """Estimates of the OR using simple logistic regression.

    Assumes no interaction between Z and X in the logistic model.
    """
    Z_X_samples = np.concatenate((Z_samples.reshape(-1,1), X_samples), axis=1)
    clf = LogisticRegression(random_state=0, solver='lbfgs').fit(Z_X_samples, Y_samples)
    return math.exp(clf.coef_[0][0])
    
def get_stratified_OR_estimate(Z_samples, X_samples, Y_samples, X_stratum_value):
    """Estimates of the OR using logistic regression stratified on X.

    Performs logistic regression for the stratum with X = X_stratum_value.
    """
    Z_X_samples = np.concatenate((Z_samples.reshape(-1,1), X_samples), axis=1)
    stratum_indices = get_stratum_indices(X_samples, X_stratum_value)
    clf = LogisticRegression(random_state=0, solver='lbfgs').fit(Z_X_samples[stratum_indices], Y_samples[stratum_indices])
    return math.exp(clf.coef_[0][0])

def get_ATE_estimate_strata(X_strata, Z_samples, X_samples, Y_samples):
    """Estimates the ATE nonparametrically by stratification on X."""
    ATE_hat = 0
    for X_stratum_value in X_strata:
        stratum_indices = get_stratum_indices(X_samples, X_stratum_value)
        pi = float(len(stratum_indices[0])) / float(len(X_samples))
        X_samples_stratum = X_samples[stratum_indices]
        Z_samples_stratum = Z_samples[stratum_indices]
        Y_samples_stratum = Y_samples[stratum_indices]

        # indices where X = X_stratum_value and Z = 1
        stratum_indices_1 = np.where(Z_samples_stratum)[0]
        Y_hat_1 = np.sum(Y_samples_stratum[stratum_indices_1]) / float(len(stratum_indices_1))

        # indices where X = X_stratum_value and Z = 0
        stratum_indices_0 = np.where(np.ones_like(Z_samples_stratum) - Z_samples_stratum)[0]
        Y_hat_0 = np.sum(Y_samples_stratum[stratum_indices_0]) / float(len(stratum_indices_0))

        ATE_hat_stratum = Y_hat_1 - Y_hat_0
        ATE_hat += pi * ATE_hat_stratum
    return ATE_hat

def get_ATE_estimate_regression(X_strata, Z_samples, X_samples, Y_samples):
    """Estimates the ATE parametrically using logistic regression."""
    Z_X_samples = np.concatenate((Z_samples.reshape(-1,1), X_samples), axis=1)
    ATE_hat = 0
    for X_stratum_value in X_strata:
        stratum_indices = get_stratum_indices(X_samples, X_stratum_value)
        pi = float(len(stratum_indices[0])) / float(len(X_samples))
        clf = LogisticRegression(random_state=0, solver='lbfgs').fit(Z_X_samples[stratum_indices], Y_samples[stratum_indices])
        beta_0_x = clf.intercept_[0]
        beta_1_x = clf.coef_[0][0]
        Y_hat_1 = expit(beta_0_x + beta_1_x)
        Y_hat_0 = expit(beta_0_x)
        ATE_hat_stratum = Y_hat_1 - Y_hat_0
        ATE_hat += pi * ATE_hat_stratum
    return ATE_hat
