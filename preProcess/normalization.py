from sklearn.preprocessing import MinMaxScaler, StandardScaler, Normalizer
from sklearn import preprocessing
import numpy as np

from .fingerprint_trucate import WavenumberTruncator

class Normalization:
    def vector_norm(self, X: np.ndarray):
        normalizer = Normalizer(norm='l2')
        return normalizer.fit_transform(X)
    
    def min_max_scaling(self, X: np.ndarray):
        min_max_scaler = MinMaxScaler()
        return min_max_scaler.fit_transform(X)
    
    def standard_scaling(self, X: np.ndarray):
        standard_scaler = StandardScaler()
        return standard_scaler.fit_transform(X)
    
    def mean_normalization(self, X: np.ndarray):
        mean = X.mean(axis=0)
        range_ = X.max(axis=0) - X.min(axis=0)
        return (X - mean) / range_
    
    def peak_normalization(self, X: np.ndarray, lower_bound:int, upper_bound:int):
        """
        Normalize each sample by its higher value within the specified wavenumber range.
        """
        waveNumber = WavenumberTruncator()
        peak_band = waveNumber.trucate_range(X, lower_bound, upper_bound)
        peak_val = np.max(peak_band, axis=1).reshape(-1,1)
        return X / peak_val