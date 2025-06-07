"""
Wave-Gaussian Mixture Prior for Adaptive ARD

This implementation introduces a novel prior that combines wave-like oscillations
with Gaussian mixtures to better capture complex patterns in building energy data.
The prior is particularly suited for capturing:
1. Periodic patterns in energy consumption
2. Multi-modal distributions of building characteristics
3. Complex interactions between features
4. Non-linear relationships in the data

Key innovations:
1. Wave component for capturing periodic patterns
2. Gaussian mixture for multi-modal distributions
3. Adaptive wave parameters based on data characteristics
4. Dynamic mixture weights
5. Feature-specific wave frequencies

Author: George Paul
Institution: The University of Bath
"""

import numpy as np
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import logging
from scipy.stats import norm
from scipy.special import digamma, polygamma
import matplotlib.pyplot as plt

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class WaveGaussianPriorConfig:
    """
    Configuration for the Wave-Gaussian Mixture Prior.
    
    Attributes:
        n_components: Number of Gaussian components in the mixture
        n_waves: Number of wave components
        wave_frequencies: Base frequencies for wave components
        wave_amplitudes: Initial amplitudes for wave components
        mixture_weights: Initial weights for Gaussian components
        adaptation_rate: Rate at which prior parameters adapt
        wave_adaptation: Whether to adapt wave parameters
        mixture_adaptation: Whether to adapt mixture weights
        numerical_stability: Small constant for numerical stability
    """
    n_components: int = 3
    n_waves: int = 2
    wave_frequencies: List[float] = None
    wave_amplitudes: List[float] = None
    mixture_weights: List[float] = None
    adaptation_rate: float = 0.1
    wave_adaptation: bool = True
    mixture_adaptation: bool = True
    numerical_stability: float = 1e-10

    def __post_init__(self):
        if self.wave_frequencies is None:
            self.wave_frequencies = [0.5, 1.0]  # Base frequencies
        if self.wave_amplitudes is None:
            self.wave_amplitudes = [0.1, 0.2]  # Initial amplitudes
        if self.mixture_weights is None:
            self.mixture_weights = [1/self.n_components] * self.n_components

class WaveGaussianPrior:
    """
    Implementation of the Wave-Gaussian Mixture Prior.
    
    This prior combines:
    1. Wave components for capturing periodic patterns
    2. Gaussian mixture for multi-modal distributions
    3. Adaptive parameters based on data characteristics
    
    The prior is particularly effective for:
    - Building energy data with seasonal patterns
    - Multi-modal distributions of building characteristics
    - Complex feature interactions
    - Non-linear relationships
    """
    
    def __init__(self, config: Optional[WaveGaussianPriorConfig] = None):
        """
        Initialize the Wave-Gaussian Mixture Prior.
        
        Args:
            config: Optional configuration object
        """
        self.config = config or WaveGaussianPriorConfig()
        
        # Initialize prior parameters
        self.wave_params = {
            'frequencies': np.array(self.config.wave_frequencies),
            'amplitudes': np.array(self.config.wave_amplitudes),
            'phases': np.zeros(self.config.n_waves)
        }
        
        self.mixture_params = {
            'weights': np.array(self.config.mixture_weights),
            'means': np.zeros(self.config.n_components),
            'variances': np.ones(self.config.n_components)
        }
        
        # Feature-specific parameters
        self.feature_params = {}
        
    def _wave_component(self, x: np.ndarray, feature_idx: int) -> np.ndarray:
        """
        Compute the wave component of the prior.
        
        Args:
            x: Input values
            feature_idx: Index of the feature
            
        Returns:
            Wave component values
        """
        if feature_idx not in self.feature_params:
            self.feature_params[feature_idx] = {
                'frequencies': self.wave_params['frequencies'].copy(),
                'amplitudes': self.wave_params['amplitudes'].copy(),
                'phases': self.wave_params['phases'].copy()
            }
            
        params = self.feature_params[feature_idx]
        wave = np.zeros_like(x)
        
        for i in range(self.config.n_waves):
            wave += params['amplitudes'][i] * np.sin(
                2 * np.pi * params['frequencies'][i] * x + params['phases'][i]
            )
            
        return wave
        
    def _gaussian_mixture(self, x: np.ndarray) -> np.ndarray:
        """
        Compute the Gaussian mixture component of the prior.
        
        Args:
            x: Input values
            
        Returns:
            Gaussian mixture values
        """
        mixture = np.zeros_like(x)
        
        for i in range(self.config.n_components):
            mixture += self.mixture_params['weights'][i] * norm.pdf(
                x,
                loc=self.mixture_params['means'][i],
                scale=np.sqrt(self.mixture_params['variances'][i])
            )
            
        return mixture
        
    def log_prior(self, w: np.ndarray, feature_idx: int) -> float:
        """
        Compute the log of the prior probability.
        
        Args:
            w: Weight values
            feature_idx: Index of the feature
            
        Returns:
            Log prior probability
        """
        # Wave component
        wave = self._wave_component(w, feature_idx)
        
        # Gaussian mixture component
        mixture = self._gaussian_mixture(w)
        
        # Combine components with numerical stability
        prior = np.clip(mixture * (1 + wave), self.config.numerical_stability, None)
        
        return np.sum(np.log(prior))
        
    def update_parameters(self, w: np.ndarray, feature_idx: int, iteration: int):
        """
        Update prior parameters based on current weights.
        
        Args:
            w: Current weight values
            feature_idx: Index of the feature
            iteration: Current iteration number
        """
        if self.config.wave_adaptation:
            self._update_wave_parameters(w, feature_idx)
            
        if self.config.mixture_adaptation:
            self._update_mixture_parameters(w)
            
    def _update_wave_parameters(self, w: np.ndarray, feature_idx: int):
        """
        Update wave parameters based on current weights.
        
        Args:
            w: Current weight values
            feature_idx: Index of the feature
        """
        params = self.feature_params[feature_idx]
        
        # Update frequencies based on weight distribution
        weight_fft = np.fft.fft(w)
        freqs = np.fft.fftfreq(len(w))
        dominant_freq = freqs[np.argmax(np.abs(weight_fft))]
        
        # Update parameters with adaptation rate
        params['frequencies'] = (
            (1 - self.config.adaptation_rate) * params['frequencies'] +
            self.config.adaptation_rate * np.abs(dominant_freq)
        )
        
        # Update amplitudes based on weight variance
        weight_std = np.std(w)
        params['amplitudes'] = (
            (1 - self.config.adaptation_rate) * params['amplitudes'] +
            self.config.adaptation_rate * weight_std
        )
        
        # Update phases to align with weight patterns
        params['phases'] = (
            (1 - self.config.adaptation_rate) * params['phases'] +
            self.config.adaptation_rate * np.angle(weight_fft[np.argmax(np.abs(weight_fft))])
        )
        
    def _update_mixture_parameters(self, w: np.ndarray):
        """
        Update Gaussian mixture parameters based on current weights.
        
        Args:
            w: Current weight values
        """
        # Update means using weighted average
        for i in range(self.config.n_components):
            self.mixture_params['means'][i] = (
                (1 - self.config.adaptation_rate) * self.mixture_params['means'][i] +
                self.config.adaptation_rate * np.mean(w)
            )
            
        # Update variances using weighted variance
        for i in range(self.config.n_components):
            self.mixture_params['variances'][i] = (
                (1 - self.config.adaptation_rate) * self.mixture_params['variances'][i] +
                self.config.adaptation_rate * np.var(w)
            )
            
        # Update mixture weights based on component responsibilities
        responsibilities = np.zeros(self.config.n_components)
        for i in range(self.config.n_components):
            responsibilities[i] = np.sum(
                self.mixture_params['weights'][i] * norm.pdf(
                    w,
                    loc=self.mixture_params['means'][i],
                    scale=np.sqrt(self.mixture_params['variances'][i])
                )
            )
            
        # Normalize responsibilities
        responsibilities = np.clip(responsibilities, self.config.numerical_stability, None)
        responsibilities /= np.sum(responsibilities)
        
        # Update weights with adaptation rate
        self.mixture_params['weights'] = (
            (1 - self.config.adaptation_rate) * self.mixture_params['weights'] +
            self.config.adaptation_rate * responsibilities
        )
        
    def visualize_prior(self, feature_idx: int, save_path: Optional[str] = None):
        """
        Visualize the prior distribution for a specific feature.
        
        Args:
            feature_idx: Index of the feature
            save_path: Optional path to save the visualization
        """
        x = np.linspace(-3, 3, 1000)
        
        # Compute components
        wave = self._wave_component(x, feature_idx)
        mixture = self._gaussian_mixture(x)
        prior = mixture * (1 + wave)
        
        # Create visualization
        plt.figure(figsize=(12, 8))
        
        # Plot components
        plt.subplot(2, 1, 1)
        plt.plot(x, wave, label='Wave Component')
        plt.plot(x, mixture, label='Gaussian Mixture')
        plt.title('Prior Components')
        plt.legend()
        
        # Plot combined prior
        plt.subplot(2, 1, 2)
        plt.plot(x, prior, label='Combined Prior')
        plt.title('Wave-Gaussian Mixture Prior')
        plt.legend()
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path)
        plt.close()
        
    def get_prior_parameters(self) -> Dict:
        """
        Get current prior parameters.
        
        Returns:
            Dictionary of prior parameters
        """
        return {
            'wave_params': self.wave_params,
            'mixture_params': self.mixture_params,
            'feature_params': self.feature_params
        }
        
    def set_prior_parameters(self, params: Dict):
        """
        Set prior parameters.
        
        Args:
            params: Dictionary of prior parameters
        """
        self.wave_params = params['wave_params']
        self.mixture_params = params['mixture_params']
        self.feature_params = params['feature_params'] 