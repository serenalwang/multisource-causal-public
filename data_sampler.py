import numpy as np
import pandas as pd
import sklearn
from sklearn.linear_model import LogisticRegression
import tensorflow as tf


class DataSampler:
    """Samples data from an augmented version of the flu dataset."""
    
    def __init__(self, z_column, x_columns, y_column):
        self.z_column = z_column
        self.x_columns = x_columns
        self.y_column = y_column
        
    def fit_outcome(self, df_input):
        """Fits outcome model P(Y = 1 | X = x, Z = z) using df_intput."""
        raise("Not Implemented")
    
    def predict_Ys(self, df_input):
        """Returns a numpy array of predictions for each row in df_input."""
        raise("Not Implemented")
    
    def fit_propensity(self, df_input, update_internal=True, print_results=False):
        """Fits logistic regression model on Z given X.
        
        If update_internal is True, updates internal propensity scoring model.
        
        """
        inputs = df_input[self.x_columns]
        clf_propensity = LogisticRegression(random_state=0, solver='lbfgs', max_iter=10000, penalty='none').fit(inputs, df_input[self.z_column])
        if print_results:
            print("Training accuracy for propensity model: %f" % clf_propensity.score(inputs, df_input[self.z_column]))
        if update_internal:
            self.clf_propensity = clf_propensity
        return clf_propensity    

    def predict_Zs(self, df_input):
        Z_probs = self.clf_propensity.predict_proba(df_input[self.x_columns]).T[1]
        Z = np.random.binomial(1, p=Z_probs)
        return Z
    
    def generate_new_data(self, df_input, n_samples):
        """Generate new data based on trained models starting with df_input."""
        # Generate X's by sampling with replacement from empirical distribution.
        new_df = df_input[self.x_columns].sample(n_samples, replace=True)
        # Generate Z's probabilistically.
        new_df[self.z_column] = self.predict_Zs(new_df)
        # Generate Y's probabilistically.
        new_df[self.y_column] = self.predict_Ys(new_df)
        return new_df
    
    def selection_bias_filter(self, df_input, p0=0.1, p1=0.9):
        """Filter the input Z_samples, X_samples, Y_samples according 
        selection bias given by and P(S = 1 | Y = 0) = p0 and P(S = 1 | Y = 1) = p1.
        """
        S_probs = p1*df_input[self.y_column] + p0*(1-df_input[self.y_column])
        S = np.random.binomial(1, p=S_probs)
        return df_input[S > 0]
    
    def generate_selection_biased_data(self, df_input, num_samples, p0=0.1, p1=0.9):
        """Returns dataframe containing data generated with selection bias.
        
        Seeded with df_input.
        
        """
        new_df = self.generate_new_data(df_input, num_samples*10)
        print("Generated %d samples before selection bias" % len(new_df))
        new_df = self.selection_bias_filter(new_df, p0=p0, p1=p1)
        print("Filtered to %d samples after selection bias; only returning the requested %d" % (len(new_df), num_samples))
        assert(len(new_df) >= num_samples)
        return new_df.iloc[:num_samples]

    def get_ATE_estimate(self, df_input):
        df_treatment = df_input.copy()
        df_treatment.loc[:,self.z_column] = 1
        y_treatment = self.predict_Ys(df_treatment, probs=True)

        df_treatment.loc[:,self.z_column] = 0
        y_control = self.predict_Ys(df_treatment, probs=True)
        ATE_estimate = np.mean(y_treatment - y_control)
        return ATE_estimate
    
    def compute_kernel(self, x_1, x_2, bandwidth=None):
        '''
        compute the kernel function value given two x_i s and a bandwidth
        Gaussian kernel.
        '''
        kernel_value= np.exp((-(np.linalg.norm(x_1-x_2)**2))/(2*bandwidth**2))
        return kernel_value
    
    def get_conditional_OR_estimates_kernel(self, input_df=None, x_inputs=None, bandwidth=None, exp=False):
        """Gets a vector of conditional OR estimates given a vector of x_inputs and a given df.

        Uses kernel estimator. This method is non parametric and does not depend on the models.

        Args:
          input_df: input dataframe
          x_inputs: dataframe where each row represents a vector x for which to estimate OR(x).

        Returns:
          OR_estimates: vector of conditional OR estimates, OR(x).

        """
#         print('compute kernel OR estimates')
        OR_estimates = []
        for x_input in np.array(x_inputs):
            kernels_x = np.array([self.compute_kernel(x_1=x_input, x_2=x_i, bandwidth=bandwidth) for x_i in np.array(input_df[self.x_columns])])
            w_1_all = np.multiply(input_df[self.y_column], input_df[self.z_column])
            w_2_all = np.multiply(np.subtract(1,input_df[self.y_column]), np.subtract(1,input_df[self.z_column]))
            w_3_all = np.multiply(input_df[self.y_column], np.subtract(1,input_df[self.z_column]))
            w_4_all = np.multiply(np.subtract(1,input_df[self.y_column]), input_df[self.z_column])
            w_all = [w_1_all, w_2_all, w_3_all, w_4_all]
            w_all = [np.average(w, weights = kernels_x) for w in w_all]
            OR_estimates.append((w_all[0]*w_all[1])/(w_all[2]*w_all[3]))
          
        OR_estimates = np.array(OR_estimates)
        if exp:         
            OR_estimates = np.exp(OR_estimates)
        return OR_estimates


class DataSamplerSimpleLogistic(DataSampler):
    """Augments the flu dataset using a simple logistic model."""
    
    def __init__(self, z_column, x_columns, y_column):
        super().__init__(z_column, x_columns, y_column)
    
    def get_input_columns(self, df_input):
        return df_input[[self.z_column] + self.x_columns]
        
    def fit_outcome(self, df_input, update_internal=True, print_results=False):
        """Fits logistic regression model on Y given X, Z.
        
        If update_internal is True, updates internal outcome model.
        
        """
        inputs = self.get_input_columns(df_input)
        clf_outcome = LogisticRegression(random_state=0, solver='lbfgs', max_iter=10000, penalty='none').fit(inputs, df_input[self.y_column])        
        if update_internal:
            self.clf_outcome = clf_outcome
        if print_results:
            y_true = df_input[self.y_column]
            print("Accuracy for outcome model: %f" % clf_outcome.score(inputs, df_input[self.y_column]))
            y_pred_probs = self.predict_Ys(df_input, probs=True)
            AUC = sklearn.metrics.roc_auc_score(y_true=y_true, y_score=y_pred_probs)
            print("AUC for outcome model: %f" % AUC)
            print('Coefficients for outcome model:', clf_outcome.coef_)
        return clf_outcome

    def print_metrics(self, df_input):
        y_true = df_input[self.y_column]
        inputs = inputs = self.get_input_columns(df_input)
        print("Accuracy for outcome model: %f" % self.clf_outcome.score(inputs, df_input[self.y_column]))
        y_pred_probs = self.predict_Ys(df_input, probs=True)
        AUC = sklearn.metrics.roc_auc_score(y_true=y_true, y_score=y_pred_probs)
        print("AUC for outcome model: %f" % AUC)
        
    def predict_Ys(self, df_input, probs=False):
        """Predicts outcome Y for given df_input.
        
        If probs is False, returns binary predictions for Y. 
        If probs is True, returns probabilities.
        
        """
        inputs = self.get_input_columns(df_input)
        Y_probs = self.clf_outcome.predict_proba(inputs).T[1]
        if probs:
            return Y_probs
        else:
            Y = np.random.binomial(1, p=Y_probs)
            return Y 


class DataSamplerInteractionLogistic(DataSamplerSimpleLogistic):
    """Augments the flu dataset using a logistic model with interaction terms."""
    
    def __init__(self, z_column, x_columns, y_column):
        super().__init__(z_column, x_columns, y_column)
    
    def get_input_columns(self, df_input, deep_copy=False):
        inputs = df_input[[self.z_column] + self.x_columns].copy(deep=deep_copy)
        for x_column in self.x_columns:
            inputs[x_column + '*Z'] = inputs[x_column] * inputs[self.z_column]
        return inputs
    
    def get_conditional_OR_estimates(self, x_inputs, exp=False):
        """Gets a vector of conditional OR estimates given a vector of x_inputs.

        Uses internal trained logistic model.

        Args:
          x_inputs: dataframe where each row represents a vector x for which to estimate OR(x).

        Returns:
          OR_estimates: vector of conditional OR estimates, OR(x).

        """
        coefs = self.clf_outcome.coef_[0]
        z_coef, x_coef, xz_coef = np.split(coefs, [1, 1 + len(self.x_columns)])
        OR_estimates = z_coef + np.dot(x_inputs, xz_coef)
        if exp:         
            OR_estimates = np.exp(OR_estimates)
        return OR_estimates


class DataSamplerNN(DataSampler):
    """Augments the flu dataset using a neural network model."""
    
    def __init__(self, z_column, x_columns, y_column):
        super().__init__(z_column, x_columns, y_column)
    
    def get_input_columns(self, df_input):
        return [df_input[self.x_columns].values.astype(float), df_input[self.z_column].values.astype(float)]
    
    def build_dnn(self, num_hidden_layers=1, hidden_dim=10):
        x_layer = tf.keras.layers.Input(shape=(len(self.x_columns),))
        z_layer = tf.keras.layers.Input(shape=(1,))
        input_layer = x_layer 

        # Add hidden layers
        for i in range(num_hidden_layers):
            new_input_layer = tf.keras.layers.Dense(
                units=hidden_dim,
                activation='relu'
            )(input_layer)
            input_layer = new_input_layer

        # Add beta layers
        output_layer = tf.keras.layers.Dense(
            units=2,
            activation=None,
            name='beta_x'
        )(input_layer)

        keras_model = tf.keras.models.Model(inputs=[x_layer, z_layer], outputs=output_layer)
        return keras_model
    
    def train_dnn(self, input_model, df_input, step_size=0.01, epochs=100, batch_size=None, verbose=1):
        model_inputs = input_model.inputs
        model_output = input_model.output
        stacked_output = tf.keras.layers.Lambda(lambda inputs: tf.concat(inputs, axis=1))([model_output, model_inputs[-1]])
        model = tf.keras.Model(inputs=model_inputs, outputs=stacked_output)

        def nn_loss(y_true, stacked_output):
            # stacked_output contains beta_x outputs and z.
            beta_x_0, beta_x_1, z = tf.split(stacked_output, 3, axis=1)
            logits = beta_x_0 + beta_x_1 * z
            return tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=y_true, logits=logits))

        model.compile(loss=nn_loss, optimizer=tf.keras.optimizers.Adam(step_size))
        model.fit(self.get_input_columns(df_input), 
                  tf.convert_to_tensor(df_input[self.y_column], dtype=tf.float32), 
                  epochs=epochs, 
                  batch_size=batch_size, 
                  verbose=verbose)
        return input_model
        
    def fit_outcome(self, df_input, num_hidden_layers=1, hidden_dim=10, step_size=0.01, epochs=100, batch_size=None, verbose=1, print_metrics=True):
        """Fits logistic regression model on Y given X, Z.
        
        If update_internal is True, updates internal outcome model.
        
        """
        tf.keras.backend.clear_session()
        beta_x_model = self.build_dnn(num_hidden_layers=num_hidden_layers, hidden_dim=num_hidden_layers)
        beta_x_model = self.train_dnn(beta_x_model, 
                                      df_input, 
                                      step_size=step_size, 
                                      epochs=epochs, 
                                      batch_size=batch_size, 
                                      verbose=verbose)
        self.beta_x_model = beta_x_model
        if print_metrics:
            self.print_metrics(df_input)

    def print_metrics(self, df_input):
        y_true = df_input[self.y_column]
        y_pred_probs = self.predict_Ys(df_input, probs=True)
        y_pred = (y_pred_probs > 0.5).numpy().astype(float)
        accuracy = sklearn.metrics.accuracy_score(y_true=y_true, y_pred=y_pred)
        print("Accuracy for outcome model: %f" % accuracy)
        AUC = sklearn.metrics.roc_auc_score(y_true=y_true, y_score=y_pred_probs)
        print("AUC for outcome model: %f" % AUC)
        return accuracy
        
    def predict_Ys(self, df_input, probs=False):
        """Predicts outcome Y for given df_input.
        
        If probs is False, returns binary predictions for Y. 
        If probs is True, returns probabilities.
        
        """
        inputs = self.get_input_columns(df_input)
        beta_x = self.beta_x_model(inputs) 
        beta_x_0, beta_x_1 = tf.split(beta_x, 2, axis=1)
        
        z = df_input[self.z_column].values.astype(float)
        logits = beta_x_0 + beta_x_1 * z.reshape((-1,1))
        Y_probs = tf.sigmoid(logits)
        if probs:
            return Y_probs
        else:
            Y = np.random.binomial(1, p=Y_probs)
            return Y 

    def get_conditional_OR_estimates(self, x_inputs, exp=False):
        """Gets a vector of conditional OR estimates given a vector of x_inputs.

        Uses internal trained logistic model.

        Args:
          x_inputs: dataframe where each row represents a vector x for which to estimate OR(x).

        Returns:
          OR_estimates: vector of conditional OR estimates, OR(x).

        """
        zs_placeholder = np.zeros(len(x_inputs))
        inputs = [x_inputs.values.astype(float), zs_placeholder]
        beta_x = self.beta_x_model(inputs) 
        beta_x_0, beta_x_1 = tf.split(beta_x, 2, axis=1)
        OR_estimates = beta_x_1
        if exp:         
            OR_estimates = np.exp(OR_estimates)
        return OR_estimates.numpy().flatten()
