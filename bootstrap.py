"""Functions for bootstrap estimates of variance."""

import numpy as np
import estimators

def run_bootstrap_df(df_obs, 
                     df_bias,  
                     n_replicates,
                     ATE_estimator_fn,
                     CV_estimator_fn,
                     optimal_CV_coeff=None,
                     random_state=42):
    """Runs bootstrap to get replicates of estimators for data in dataframes.
    
    Args:
      df_obs: dataframe containing samples from observational dataset.
      df_bias: dataframe containing samples from selection bias dataset.
      n_replicates: number of bootstrap replicates.
      ATE_estimator_fn: Function which returns an ATE estimate given an input dataframe.
        Args: df_input
        Returns: float ATE estimate. 
      CV_estimator_fn: Function which returns a control variate estimate given two input dataframes.
        Args: df_input_obs (observational dataset), df_input_bias (selection bias dataset)
        Returns: float or numpy array of control variate estimates.
      optimal_CV_coeff: Coefficient to use when subtracting the control variates. 
        If None, then ATE_hat_CV_samples will be returned empty. 

    Returns:
      CV_samples: vector of replicates of the control variate estimator.
      ATE_hat_samples: vector of replicates of the ATE estimator.
      ATE_hat_CV_samples: vector of replicates of the ATE estimator with control variates subtracted.
        Empty if optimal_CV_coeff is None.
    """

    CV_samples = []
    ATE_hat_samples = []
    ATE_hat_CV_samples = []
    for i in range(n_replicates):
        print("Starting replicate %d" % i, end='\r')
        # bootstrap sample from observational dataset
        df_obs_bs = df_obs.sample(len(df_obs), replace=True, random_state=random_state + i)
        
        # Get ATE estimate
        ATE_hat = ATE_estimator_fn(df_obs_bs)
        ATE_hat_samples.append(ATE_hat)

        # bootstrap sample from selection bias dataset
        df_bias_bs = df_bias.sample(len(df_bias), replace=True, random_state=random_state + i)
        
        # Get OR estimates and compute the control variate
        CV_val = CV_estimator_fn(df_obs_bs, df_bias_bs)
        CV_samples.append(CV_val)
        
        # Add the control variate to the ATE estimate.
        if optimal_CV_coeff is not None:
            ATE_hat_CV = ATE_hat - np.dot(optimal_CV_coeff, CV_val)
            ATE_hat_CV_samples.append(ATE_hat_CV)
            
    return CV_samples, ATE_hat_samples, ATE_hat_CV_samples

def run_bootstrap_np(Z_samples_obs, 
                  X_samples_obs, 
                  Y_samples_obs, 
                  Z_samples_bias, 
                  X_samples_bias, 
                  Y_samples_bias,
                  X_strata, 
                  n_replicates,
                  ATE_estimator_type='strata',
                  optimal_CV_coeff=None, 
                  CV_stratified=False):
    """Runs bootstrap to get replicates of estimators for data in np arrays.
    
    Args:
      Z_samples_obs, X_samples_obs, Y_samples_obs: samples from observational dataset.
      Z_samples_bias, X_samples_bias, Y_samples_bias: samples from selection bias dataset.
      X_strata: possible discrete values for X.
      n_replicates: number of bootstrap replicates.
      ATE_estimator_type: if 'strata', use the stratified estimator. If 'reg', use regression estimator.
      optimal_CV_coeff: Coefficient to use when subtracting the control variates. 
        If None, then ATE_hat_CV_samples will be returned empty. 
      CV_stratified: If True, computes vector of control variates with stratification.

    Returns:
      CV_samples: vector of replicates of the control variate estimator, OR_O1 - OR_O2.
      ATE_hat_samples: vector of replicates of the ATE estimator.
      ATE_hat_CV_samples: vector of replicates of the ATE estimator with control variates subtracted.
        Empty if optimal_CV_coeff is None.
    """

    CV_samples = []
    ATE_hat_samples = []
    ATE_hat_CV_samples = []
    for _ in range(n_replicates):
        bs_indices_obs = np.random.choice(np.arange(len(Y_samples_obs)), replace=True, size=len(Y_samples_obs))
        Z_samples_obs_bs = Z_samples_obs[bs_indices_obs]
        X_samples_obs_bs = X_samples_obs[bs_indices_obs]
        Y_samples_obs_bs = Y_samples_obs[bs_indices_obs]
        
        # Get ATE estimate
        if ATE_estimator_type == 'strata':
            ATE_hat = estimators.get_ATE_estimate_strata(X_strata, Z_samples_obs_bs, X_samples_obs_bs, Y_samples_obs_bs)
        elif ATE_estimator_type == 'reg':
            ATE_hat = estimators.get_ATE_estimate_regression(X_strata, Z_samples_obs_bs, X_samples_obs_bs, Y_samples_obs_bs)
        else:
            raise("ATE_estimator_type not recognized.")
        ATE_hat_samples.append(ATE_hat)

        bs_indices_bias = np.random.choice(np.arange(len(Y_samples_bias)), replace=True, size=len(Y_samples_bias))
        Z_samples_bias_bs = Z_samples_bias[bs_indices_bias]
        X_samples_bias_bs = X_samples_bias[bs_indices_bias]
        Y_samples_bias_bs = Y_samples_bias[bs_indices_bias]
        OR_vec_obs = []
        OR_vec_bias = []
        
        # Get OR estimates and compute the control variate
        CV_val = None
        if CV_stratified: 
            for X_stratum_value in X_strata:
                strat_OR_estimate_obs = estimators.get_stratified_OR_estimate(Z_samples_obs_bs, X_samples_obs_bs, Y_samples_obs_bs, X_stratum_value)
                OR_vec_obs.append(strat_OR_estimate_obs)
                strat_OR_estimate_bias = estimators.get_stratified_OR_estimate(Z_samples_bias_bs, X_samples_bias_bs, Y_samples_bias_bs, X_stratum_value)
                OR_vec_bias.append(strat_OR_estimate_bias)
            CV_val = np.array(OR_vec_obs) - np.array(OR_vec_bias)
        else: 
            unstrat_OR_estimate_obs = estimators.get_unstratified_OR_estimate(Z_samples_obs_bs, X_samples_obs_bs, Y_samples_obs_bs)   
            unstrat_OR_estimate_bias = estimators.get_unstratified_OR_estimate(Z_samples_bias_bs, X_samples_bias_bs, Y_samples_bias_bs)
            CV_val = unstrat_OR_estimate_obs - unstrat_OR_estimate_bias
        CV_samples.append(CV_val)
        
        # Add the control variate to the ATE estimate.
        if optimal_CV_coeff is not None:
            ATE_hat_CV = ATE_hat - np.dot(optimal_CV_coeff, CV_val)
            ATE_hat_CV_samples.append(ATE_hat_CV)
            
    return CV_samples, ATE_hat_samples, ATE_hat_CV_samples