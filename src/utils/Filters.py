import numpy as np
import statsmodels.api as sm

class ARFilter:
    def __init__(self, X: np.ndarray, max_p: int = 10, ic: str = 'aic', best= False):
        """
        Initializes the ARFilter with a matrix X and fits an AR model to each column.
        Parameters:
        X : np.ndarray
            Input data matrix of shape (n, d), where each column is a time series.
        best: Boolean
            If true, best model is selected, otherwise p is set to max_p
        max_p : int
            Maximum lag order to consider when selecting the best AR model. 
        ic : str
            Information criterion to use for lag selection: 'aic' or 'bic'.
        """
        self.data = X.copy()
        self.models = []
        self.max_p = max_p
        self.p_orders = []
        n, D = self.data.shape

        # Determine optimal lag and fit model for each dimension
        self.residuals = np.empty((n, D))

        if n!= 0:
            for j in range(D):
                best_model = None
                best_ic = np.inf
                best_p = self.max_p
                if best is False:
                    best_model = sm.tsa.AutoReg(self.data[:, j], lags=best_p, old_names=False).fit()
                    model_ic = getattr(best_model, ic)
                else:
                    for p in range(1, max_p + 1):
                        try:
                            model = sm.tsa.AutoReg(self.data[:, j], lags=p, old_names=False).fit()
                            model_ic = getattr(model, ic)
                            if model_ic < best_ic:
                                best_ic = model_ic
                                best_model = model
                                best_p = p
                        except Exception:
                            continue
                self.models.append(best_model)
                self.p_orders.append(best_p)
                self.residuals[best_p:, j] = self.data[best_p:, j] - best_model.fittedvalues
                self.residuals[:best_p, j] = np.nan 

            
    def get_residuals(self):
        """Returns the residuals, not the forecast residuals."""
        return self.residuals

    def get_noise_std(self):
        """Returns the estimated standard deviation (sigma) of the residuals for each variable."""
        return np.array([np.sqrt(model.sigma2) for model in self.models])

    def predict_errors(self, new_data):
        """
        Computes one-step-ahead prediction errors for all time series using their respective AR models.

        Parameters:
            new_data : np.ndarray
                New data matrix of shape (m, d), where each column is a time series.

        Returns:
            np.ndarray
                One-step-ahead prediction errors of shape (m, d).
        """
        m, d = new_data.shape
        errors = new_data.copy()

        if len(self.models) != 0:
            for j in range(d):
                p = self.p_orders[j]
                model = self.models[j]
                coef = model.params
                has_intercept = 'const' in model.model.exog_names
                full_series = np.concatenate((self.data[-p:, j], new_data[:, j]))
                X_lagged = np.column_stack([
                    full_series[i : i + m] for i in range(p - 1, -1, -1)
                ])

                if has_intercept:
                    X_lagged = np.column_stack([np.ones(m), X_lagged]) 

                # Predict and compute errors
                pred = X_lagged @ coef
                errors[:, j] = new_data[:, j] - pred
        self.data = np.vstack((self.data, new_data))
        return errors