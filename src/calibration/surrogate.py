import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, ConstantKernel as C
from scipy.stats import norm

class KrigingSurrogate:
    def __init__(self, random_state=42):
        # 使用 Matern 核，这是空间建模中常用的核函数
        kernel = C(1.0, (1e-3, 1e3)) * Matern(length_scale=1.0, length_scale_bounds=(1e-2, 1e2), nu=1.5)
        self.model = GaussianProcessRegressor(
            kernel=kernel,
            n_restarts_optimizer=10,
            alpha=1e-2, # 观测噪声
            normalize_y=True,
            random_state=random_state
        )
        self.is_fitted = False

    def fit(self, X, y):
        """
        训练代理模型
        X: (n_samples, n_features)
        y: (n_samples, )
        """
        X = np.atleast_2d(X)
        self.model.fit(X, y)
        self.is_fitted = True

    def predict(self, X):
        """
        预测均值和标准差
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction.")
        X = np.atleast_2d(X)
        mu, sigma = self.model.predict(X, return_std=True)
        return mu, sigma

    def expected_improvement(self, X_candidates, best_y, xi=0.01):
        """
        计算期望改进量 (EI)
        X_candidates: 候选点集
        best_y: 当前已知的最小目标值 (RMSE)
        xi: 探索因子
        """
        mu, sigma = self.predict(X_candidates)
        
        # 因为我们的目标是最小化 RMSE，所以改进是 best_y - mu
        with np.errstate(divide='warn'):
            imp = best_y - mu - xi
            Z = imp / sigma
            ei = imp * norm.cdf(Z) + sigma * norm.pdf(Z)
            ei[sigma <= 0.0] = 0.0
            
        return ei
