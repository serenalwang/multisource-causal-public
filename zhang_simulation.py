"""Functions for generating data and running estimation for the simulation based on Zhang 2009."""

import numpy as np
import math
from scipy.special import expit 

import bootstrap

class zhang_sampler:
    """Samples data from a model based on Zhang 2009."""
    X_strata = [np.array([0,0]), np.array([0,1]), np.array([1,0]), np.array([1,1])]
    
    def __init__(self, a1_true, b01_true, b11_true):
        """Initialize true model parameters.
        
        The true data generating model is:
        P(Z = 1 | X = x) = expit(a0_true + <a1_true, x>)
        P(Y(0) = 1 | X = x) = expit(b00_true + <b01_true, x>)
        P(Y(1) = 1 | X = x) = expit(b10_true + <b11_true, x>)
        
        X is distributed as
        X1, X2 ~ Bern(0.5)
        """ 
        self.a1_true = a1_true
        self.X_mean = np.array([0.5,0.5]) # vector E[X]
        self.a0_true = -np.dot(self.a1_true, self.X_mean)
        self.b01_true = b01_true
        self.b00_true = -0.5 - np.dot(self.b01_true, self.X_mean)
        self.b11_true = b11_true
        self.b10_true = 0.5 - np.dot(self.b11_true, self.X_mean)
        
    def _get_X_samples(self, num_samples):
        """Returns a numpy array with shape (num_samples, 4) where each row represents a sample of X. """
        X1 = np.random.binomial(1, 0.5, num_samples)
        X2 = np.random.binomial(1, 0.5, num_samples)
        X_samples = np.array([X1,X2]).T
        return X_samples 
        
    def prob_Z_given_X(self, x):
        """ Returns P(Z = 1 | X = x), or the true propensity score.
        P(Z = 1 | X = x) = expit(a0_true + <a1_true, x>)
        """
        return expit(self.a0_true + np.dot(self.a1_true, x))
    
    def prob_Y_given_X_Z(self, x, z):
        if z == 0:
            return expit(self.b00_true + np.dot(self.b01_true, x))
        elif z == 1:
            return expit(self.b10_true + np.dot(self.b11_true, x))
        else:
            raise("z must be binary.")
    
    def get_samples(self, num_samples):
        """Returns numpy arrays Z_samples, X_samples, Y_samples."""
        X_samples = self._get_X_samples(num_samples)
        Z_samples = []
        Y_samples = []
        for x in X_samples:
            z = np.random.binomial(1, self.prob_Z_given_X(x))
            Z_samples.append(z)
            y = np.random.binomial(1, self.prob_Y_given_X_Z(x, z))
            Y_samples.append(y)
        return np.array(Z_samples), X_samples, np.array(Y_samples)
    
    def get_ATE_true(self):
        """Gets true ATE under assumption that all X strata have the same probability.""" 
        ATE_total = 0
        for X_stratum_value in self.X_strata:
            ATE_stratum = self.prob_Y_given_X_Z(X_stratum_value, 1) - self.prob_Y_given_X_Z(X_stratum_value, 0)
            ATE_total += ATE_stratum
        return ATE_total / len(self.X_strata)
    
    def selection_bias_filter(self, Z_samples, X_samples, Y_samples, p0=0.1, p1=0.9):
        """Filter the input Z_samples, X_samples, Y_samples according 
        selection bias given by and P(S = 1 | Y = 0) = p0 and P(S = 1 | Y = 1) = p1.
        """
        S_samples = [] # Binary vector where s = 1 indicates that a sample has been selected.
        for y in Y_samples:
            if y == 1: 
                s = np.random.binomial(1, p1)
            elif y == 0:
                s = np.random.binomial(1, p0)
            else:
                raise("y must be binary.")
            S_samples.append(s)
        S_indices = np.where(S_samples)
        return Z_samples[S_indices], X_samples[S_indices], Y_samples[S_indices]
    
    def get_samples_selection_bias(self, num_samples, p0=0.1, p1=0.9):
        """Returns numpy arrays Z_samples, X_samples, Y_samples with selection bias.
        
        Each array contains num_samples samples after selection bias.
        """
        Z_samples_large, X_samples_large, Y_samples_large = self.get_samples(num_samples*3)
        Z_samples_bias, X_samples_bias, Y_samples_bias = self.selection_bias_filter(Z_samples_large, X_samples_large, Y_samples_large, p0=p0, p1=p1)
        Z_samples_bias = Z_samples_bias[:num_samples]
        X_samples_bias = X_samples_bias[:num_samples]
        Y_samples_bias = Y_samples_bias[:num_samples]
        return Z_samples_bias, X_samples_bias, Y_samples_bias


def run_full_simulation(n_obs=1000, n_bias=10000, p0=0.1, p1=0.9, n_replicates=100, ATE_estimator_type='strata', CV_stratified=False):
    """Runs a full simulation with the Zhang simulation model.
    
    Args:
      n_obs: Number of samples in obserational dataset (without selection bias).
      n_bias: Number of samples in selection bias dataset (with selection bias).
      p0: probability of selecting sample when Y = 0 in O2
      p1: probability of selecting sample when Y = 1 in O2
      n_replicates: number of bootstrap replicates
      CV_stratified: if True, run simulation with b11_true = -b01_true, and compute vector of control variates. 
        Otherwise run simple logistic regression with b11_true = b01_true and scalar control variate.
        
    Returns:
      ATE_var, ATE_bias, ATE_CV_var, ATE_CV_bias: variance and bias of ATE estimator with and without CV.
    
    """
    print("Running simulation for n_obs=%d, n_bias=%d, p0=%.2f, p1=%.2f, ATE_estimator_type=%s, CV_stratified=%r" % (n_obs, n_bias, p0, p1, ATE_estimator_type, CV_stratified))
    # Simulate data
    a1_true =  np.array([1, -1])
    b01_true = np.array([1, -1])
    if CV_stratified:
        b11_true = -b01_true
    else:
        b11_true = b01_true 
    sampler = zhang_sampler(a1_true, b01_true, b11_true)
    Z_samples_obs, X_samples_obs, Y_samples_obs = sampler.get_samples(n_obs)
    Z_samples_bias, X_samples_bias, Y_samples_bias = sampler.get_samples_selection_bias(n_bias, p0=p0, p1=p1)
    
    # Get initial Cov estimates 
    CV_samples, ATE_hat_samples, _ = bootstrap.run_bootstrap_np(Z_samples_obs, X_samples_obs, Y_samples_obs,
                                                   Z_samples_bias, X_samples_bias, Y_samples_bias,
                                                   X_strata=sampler.X_strata,
                                                   ATE_estimator_type=ATE_estimator_type,
                                                   n_replicates=n_replicates, 
                                                   optimal_CV_coeff=None, 
                                                   CV_stratified=CV_stratified)
    if CV_stratified:
        CV_var = np.cov(np.array(CV_samples), ddof=1, rowvar=False)       
        sample_cov = np.cov(np.concatenate((np.array(ATE_hat_samples).reshape(1,-1),np.array(CV_samples).T)), ddof=1)
        
        # Get optimal control variates coefficient
        cov_ATE_CV = sample_cov[0][1:]
        var_CV_inv = np.linalg.inv(CV_var)
        optimal_CV_coeff = np.dot(var_CV_inv, cov_ATE_CV)

    else:
        sample_cov = np.cov(np.array([ATE_hat_samples, CV_samples]), ddof=1)

        # Get optimal control variates coefficient
        cov_ATE_CV = sample_cov[0][1]
        var_CV = sample_cov[1][1]
        optimal_CV_coeff = cov_ATE_CV / var_CV
    
    # Get variance/bias of ATE estimators with and without CV.
    CV_samples, ATE_hat_samples, ATE_hat_CV_samples = bootstrap.run_bootstrap_np(Z_samples_obs, X_samples_obs, Y_samples_obs,
                                                                    Z_samples_bias, X_samples_bias, Y_samples_bias,
                                                                    X_strata=sampler.X_strata,
                                                                    n_replicates=n_replicates,
                                                                    ATE_estimator_type=ATE_estimator_type,
                                                                    optimal_CV_coeff=optimal_CV_coeff, 
                                                                    CV_stratified=CV_stratified)
    
    ATE_var = np.var(np.array(ATE_hat_samples), ddof=1)
    print(">>> Variance of ATE estimator:", ATE_var)

    ATE_bias = np.mean(np.array(ATE_hat_samples)) - sampler.get_ATE_true()
    print(">>> Bias of ATE estimator:", ATE_bias)

    ATE_CV_var = np.var(np.array(ATE_hat_CV_samples), ddof=1)
    print(">>> Variance of ATE estimator with CV:", ATE_CV_var)

    ATE_CV_bias = np.mean(np.array(ATE_hat_CV_samples)) - sampler.get_ATE_true()
    print(">>> Bias of ATE estimator with CV:", ATE_CV_bias)
    
    return ATE_var, ATE_bias, ATE_CV_var, ATE_CV_bias