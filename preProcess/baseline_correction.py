import numpy as np
from scipy.signal import savgol_filter
from pybaselines import Baseline


class BaselineCorrection:
    def savgol_filter(self, X : np.ndarray,  window_length=13, poly_order=4, deriv=0) -> np.ndarray:
        return savgol_filter(X, window_length, poly_order, deriv=deriv)
    
    def asls_baseline(self, X : np.ndarray, lam=1e7, p=0.01, max_iter=1) -> np.ndarray:
        baseline_fitter = Baseline()

        X = np.atleast_2d(X)
        baseline = []

        for x in X:
            single_baseline, params = baseline_fitter.asls(x, lam=lam, p=p, max_iter=max_iter)
            baseline.append(single_baseline)
        
        return np.squeeze(np.array(baseline))
    
    def rubberband_baseline(self, X : np.ndarray, segments=1, lam=None, diff_order=2) -> np.ndarray:
        baseline_fitter = Baseline()

        X = np.atleast_2d(X)
        baseline = []

        for x in X:
            single_baseline, params = baseline_fitter.rubberband(x, segments=segments, lam=lam, diff_order=diff_order)
            baseline.append(single_baseline)

        return np.squeeze(np.array(baseline))
    
    def polynomial_baseline(self, X : np.ndarray, poly_order=3) -> np.ndarray:
        baseline_fitter = Baseline()

        X = np.atleast_2d(X)
        baseline = []

        for x in X:
            single_baseline, params = baseline_fitter.poly(x, poly_order=poly_order)
            baseline.append(single_baseline)

        return np.squeeze(np.array(baseline))