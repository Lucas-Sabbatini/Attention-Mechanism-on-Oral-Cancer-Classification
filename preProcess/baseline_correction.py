import numpy as np
from scipy.signal import savgol_filter
from pybaselines import Baseline


class BaselineCorrection:

    def savgol_filter(self, x : np.ndarray,  window_length=11, poly_order=3) -> np.ndarray:
        return savgol_filter(x, window_length, poly_order)
    
    def asls_baseline(self, x : np.ndarray, lam=1e6, p=0.01, max_iter=50) -> np.ndarray:
        baseline_fitter = Baseline()
        baseline, params = baseline_fitter.asls(x, lam=lam, p=p, max_iter=max_iter)
        return baseline
    
    def rubberband_baseline(self, x : np.ndarray, segments=1, lam=None, diff_order=2) -> np.ndarray:
        baseline_fitter = Baseline()
        baseline, params = baseline_fitter.rubberband(x, segments=segments, lam=lam, diff_order=diff_order)
        return baseline
    
    def polynomial_baseline(self, x : np.ndarray, poly_order=3) -> np.ndarray:
        baseline_fitter = Baseline()
        baseline, params = baseline_fitter.poly(x, poly_order=poly_order)
        return baseline