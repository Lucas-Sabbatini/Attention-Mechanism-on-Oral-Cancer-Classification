"""
Spectral Data Augmentation Pipeline

Implements synthetic data generation techniques for spectroscopy data:
1. Multiplicative Intensity Scaling
2. Additive Baseline Shift
3. Gaussian Noise Injection
4. Small Wavenumber Shift
"""

import numpy as np
from scipy.interpolate import interp1d
from typing import Optional


# Default augmentation probabilities
DEFAULT_PROBABILITIES = {
    'scaling': 0.30,
    'noise': 0.30,
    'shift': 0.20,
    'baseline': 0.10,
    'none': 0.10,
}


def multiplicative_scaling(spectrum: np.ndarray, alpha_range: tuple = (0.9, 1.1)) -> np.ndarray:
    """
    Simulates sample thickness or concentration variation.
    
    Formula: x' = α * x
    
    Args:
        spectrum: 1D array of spectral intensities
        alpha_range: Tuple (min, max) for uniform distribution of α
        
    Returns:
        Scaled spectrum
    """
    alpha = np.random.uniform(alpha_range[0], alpha_range[1])
    return spectrum * alpha


def additive_baseline_shift(spectrum: np.ndarray, beta_range: tuple = (-0.02, 0.02)) -> np.ndarray:
    """
    Simulates scattering offset.
    
    Formula: x' = x + β
    
    Args:
        spectrum: 1D array of spectral intensities
        beta_range: Tuple (min, max) for uniform distribution of β
        
    Returns:
        Shifted spectrum
    """
    beta = np.random.uniform(beta_range[0], beta_range[1])
    return spectrum + beta


def gaussian_noise_injection(
    spectrum: np.ndarray,
    sigma_percent_range: tuple = (0.005, 0.02),
    max_noise_percent: float = 0.05
) -> np.ndarray:
    """
    Simulates instrument noise.
    
    Formula: x' = x + N, where N ~ Normal(0, σ²)
    
    Args:
        spectrum: 1D array of spectral intensities
        sigma_percent_range: Tuple (min, max) for σ as percentage of mean signal amplitude
        max_noise_percent: Maximum allowed noise level (default 5%)
        
    Returns:
        Noisy spectrum
    """
    mean_amplitude = np.abs(spectrum).mean()
    
    # Sample sigma as percentage of mean amplitude
    sigma_percent = np.random.uniform(sigma_percent_range[0], sigma_percent_range[1])
    
    # Ensure noise doesn't exceed max threshold
    sigma_percent = min(sigma_percent, max_noise_percent)
    
    sigma = sigma_percent * mean_amplitude
    noise = np.random.normal(0, sigma, spectrum.shape)
    
    return spectrum + noise


def wavenumber_shift(
    spectrum: np.ndarray,
    delta_range: tuple = (-2.0, 2.0),
    wavenumber_spacing: float = 1.9286
) -> np.ndarray:
    """
    Simulates slight instrument calibration variation by shifting 
    spectrum along wavenumber axis.
    
    Args:
        spectrum: 1D array of spectral intensities
        delta_range: Tuple (min, max) for shift δ in cm⁻¹ (uniform distribution)
        wavenumber_spacing: Average spacing between wavenumbers in cm⁻¹
                           Default ~1.9286 based on wavenumbers_cancboca.dat
        
    Returns:
        Shifted and interpolated spectrum
    """
    n_points = len(spectrum)
    
    # Sample shift in cm⁻¹
    delta = np.random.uniform(delta_range[0], delta_range[1])
    
    # Convert shift to index units
    delta_index = delta / wavenumber_spacing
    
    # Original and shifted positions
    original_positions = np.arange(n_points)
    shifted_positions = original_positions + delta_index
    
    # Create interpolation function with boundary handling
    interpolator = interp1d(
        original_positions,
        spectrum,
        kind='linear',
        bounds_error=False,
        fill_value=(spectrum[0], spectrum[-1])
    )
    
    # Interpolate at shifted positions
    shifted_spectrum = interpolator(shifted_positions)
    
    return shifted_spectrum


def augment_spectrum(
    spectrum: np.ndarray,
    probabilities: Optional[dict] = None,
    scaling_params: dict = None,
    baseline_params: dict = None,
    noise_params: dict = None,
    shift_params: dict = None
) -> np.ndarray:
    """
    Apply augmentation techniques to a single spectrum with given probabilities.
    
    Multiple augmentations can be applied to the same sample based on 
    independent probability draws.
    
    Args:
        spectrum: 1D array of spectral intensities
        probabilities: Dict with keys 'scaling', 'noise', 'shift', 'baseline', 'none'
                      and corresponding probability values
        scaling_params: Dict of parameters for multiplicative_scaling
        baseline_params: Dict of parameters for additive_baseline_shift
        noise_params: Dict of parameters for gaussian_noise_injection
        shift_params: Dict of parameters for wavenumber_shift
        
    Returns:
        Augmented spectrum
    """
    if probabilities is None:
        probabilities = DEFAULT_PROBABILITIES
    
    # Initialize parameter dicts
    scaling_params = scaling_params or {}
    baseline_params = baseline_params or {}
    noise_params = noise_params or {}
    shift_params = shift_params or {}
    
    # Check if "none" case applies (no augmentation)
    if np.random.random() < probabilities.get('none', 0.10):
        return spectrum.copy()
    
    augmented = spectrum.copy()
    
    # Apply each augmentation independently based on probability
    if np.random.random() < probabilities.get('scaling', 0.30):
        augmented = multiplicative_scaling(augmented, **scaling_params)
    
    if np.random.random() < probabilities.get('baseline', 0.10):
        augmented = additive_baseline_shift(augmented, **baseline_params)
    
    if np.random.random() < probabilities.get('noise', 0.30):
        augmented = gaussian_noise_injection(augmented, **noise_params)
    
    if np.random.random() < probabilities.get('shift', 0.20):
        augmented = wavenumber_shift(augmented, **shift_params)
    
    return augmented


def augment_spectra(
    spectra: np.ndarray,
    n_augmented: Optional[int] = None,
    probabilities: Optional[dict] = None,
    scaling_params: dict = None,
    baseline_params: dict = None,
    noise_params: dict = None,
    shift_params: dict = None,
    include_original: bool = True,
    random_seed: Optional[int] = None
) -> np.ndarray:
    """
    Apply augmentation pipeline to a batch of spectra.
    
    Args:
        spectra: 2D array where rows are samples and columns are wavenumber intensities
        n_augmented: Number of augmented samples to generate per original sample.
                    If None, generates 1 augmented sample per original.
        probabilities: Dict with augmentation probabilities
        scaling_params: Parameters for multiplicative scaling
        baseline_params: Parameters for baseline shift
        noise_params: Parameters for noise injection
        shift_params: Parameters for wavenumber shift
        include_original: If True, includes original samples in output
        random_seed: Optional seed for reproducibility
        
    Returns:
        2D array with augmented (and optionally original) samples
    """
    if random_seed is not None:
        np.random.seed(random_seed)
    
    n_samples, n_features = spectra.shape
    n_augmented = n_augmented or 1
    
    # Pre-allocate output array
    if include_original:
        total_samples = n_samples * (1 + n_augmented)
        output = np.zeros((total_samples, n_features))
        output[:n_samples] = spectra  # Original samples first
        start_idx = n_samples
    else:
        total_samples = n_samples * n_augmented
        output = np.zeros((total_samples, n_features))
        start_idx = 0
    
    # Generate augmented samples
    idx = start_idx
    for i in range(n_samples):
        for _ in range(n_augmented):
            output[idx] = augment_spectrum(
                spectra[i],
                probabilities=probabilities,
                scaling_params=scaling_params,
                baseline_params=baseline_params,
                noise_params=noise_params,
                shift_params=shift_params
            )
            idx += 1
    
    return output


def augment_with_labels(
    spectra: np.ndarray,
    labels: np.ndarray,
    n_augmented: int = 1,
    probabilities: Optional[dict] = None,
    include_original: bool = True,
    random_seed: Optional[int] = None,
    **augmentation_params
) -> tuple:
    """
    Augment spectra while preserving labels.
    
    Args:
        spectra: 2D array of spectral data
        labels: 1D array of labels corresponding to spectra
        n_augmented: Number of augmented samples per original
        probabilities: Augmentation probabilities
        include_original: Include original samples in output
        random_seed: Optional seed for reproducibility
        **augmentation_params: Additional params for augmentation functions
        
    Returns:
        Tuple of (augmented_spectra, augmented_labels)
    """
    if random_seed is not None:
        np.random.seed(random_seed)
    
    n_samples = len(spectra)
    
    # Augment spectra
    augmented_spectra = augment_spectra(
        spectra,
        n_augmented=n_augmented,
        probabilities=probabilities,
        include_original=include_original,
        **augmentation_params
    )
    
    # Create corresponding labels
    if include_original:
        augmented_labels = np.concatenate([
            labels,
            np.repeat(labels, n_augmented)
        ])
    else:
        augmented_labels = np.repeat(labels, n_augmented)
    
    return augmented_spectra, augmented_labels
