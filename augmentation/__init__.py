"""
Spectral Data Augmentation Module

Provides synthetic data generation techniques for spectroscopy data:
- Multiplicative Intensity Scaling
- Additive Baseline Shift  
- Gaussian Noise Injection
- Small Wavenumber Shift
"""

from .spectral_augmentation import (
    multiplicative_scaling,
    additive_baseline_shift,
    gaussian_noise_injection,
    wavenumber_shift,
    augment_spectrum,
    augment_spectra,
    augment_with_labels,
    DEFAULT_PROBABILITIES,
)

__all__ = [
    'multiplicative_scaling',
    'additive_baseline_shift',
    'gaussian_noise_injection',
    'wavenumber_shift',
    'augment_spectrum',
    'augment_spectra',
    'augment_with_labels',
    'DEFAULT_PROBABILITIES',
]
