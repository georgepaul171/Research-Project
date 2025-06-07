# Wave-Gaussian Mixture Prior: A Novel Approach to Building Energy Modeling

## Theoretical Foundation

The Wave-Gaussian Mixture Prior is a novel Bayesian prior that combines wave-like oscillations with Gaussian mixtures to better capture complex patterns in building energy data. This prior was developed to address several key challenges in building energy modeling:

1. **Periodic Patterns**: Building energy consumption often exhibits periodic patterns (daily, weekly, seasonal)
2. **Multi-modal Distributions**: Building characteristics and energy usage often follow multi-modal distributions
3. **Feature Interactions**: Complex interactions between building features affect energy performance
4. **Non-linear Relationships**: Energy performance often has non-linear relationships with building features

### Mathematical Formulation

The prior combines two main components:

1. **Wave Component**:
   $$w(x) = \sum_{i=1}^{n_w} a_i \sin(2\pi f_i x + \phi_i)$$
   where:
   - $a_i$ are the wave amplitudes
   - $f_i$ are the frequencies
   - $\phi_i$ are the phase shifts
   - $n_w$ is the number of wave components

2. **Gaussian Mixture Component**:
   $$g(x) = \sum_{j=1}^{n_c} \pi_j \mathcal{N}(x|\mu_j, \sigma_j^2)$$
   where:
   - $\pi_j$ are the mixture weights
   - $\mu_j$ are the component means
   - $\sigma_j^2$ are the component variances
   - $n_c$ is the number of Gaussian components

The combined prior is:
$$p(x) = g(x)(1 + w(x))$$

## Key Innovations

### 1. Adaptive Wave Parameters
- Frequencies adapt based on FFT analysis of the data
- Amplitudes adjust based on feature variance
- Phases align with dominant patterns in the data

### 2. Dynamic Mixture Weights
- Weights update based on component responsibilities
- Means and variances adapt to data characteristics
- Numerical stability through clipping and normalization

### 3. Feature-Specific Adaptation
- Each feature gets its own set of wave parameters
- Adaptation rates can be feature-specific
- Parameters evolve during model training

## Implementation Details

### Prior Configuration
```python
@dataclass
class WaveGaussianPriorConfig:
    n_components: int = 3  # Number of Gaussian components
    n_waves: int = 2      # Number of wave components
    wave_frequencies: List[float] = None
    wave_amplitudes: List[float] = None
    mixture_weights: List[float] = None
    adaptation_rate: float = 0.1
```

### Parameter Updates
1. **Wave Parameters**:
   - Frequencies update based on FFT analysis
   - Amplitudes adjust based on feature variance
   - Phases align with dominant patterns

2. **Mixture Parameters**:
   - Weights update based on component responsibilities
   - Means and variances adapt to data characteristics
   - Numerical stability through clipping

## Applications in Building Energy Modeling

### 1. Feature Selection
- Wave components help identify periodic patterns
- Gaussian mixtures capture multi-modal distributions
- Combined effect improves feature relevance assessment

### 2. Uncertainty Quantification
- Wave components capture periodic uncertainty
- Gaussian mixtures model aleatoric uncertainty
- Combined effect provides comprehensive uncertainty estimates

### 3. Model Adaptation
- Prior adapts to building-specific patterns
- Parameters evolve during training
- Better captures complex relationships

## Advantages Over Traditional Priors

1. **Traditional ARD Prior**:
   - Single Gaussian component
   - Fixed shrinkage parameters
   - Limited to unimodal distributions

2. **Horseshoe Prior**:
   - Heavy-tailed but static
   - No periodic component
   - Limited adaptation

3. **Wave-Gaussian Mixture Prior**:
   - Combines periodic and multi-modal components
   - Fully adaptive parameters
   - Feature-specific adaptation

## Future Directions

1. **Theoretical Extensions**:
   - Non-stationary wave components
   - Hierarchical mixture structure
   - Cross-feature wave interactions

2. **Implementation Improvements**:
   - Parallel parameter updates
   - GPU acceleration
   - Sparse wave representations

3. **Applications**:
   - Time series forecasting
   - Anomaly detection
   - Feature interaction analysis

## References

1. MacKay, D. J. (1992). Bayesian interpolation. Neural Computation, 4(3), 415-447.
2. Carvalho, C. M., Polson, N. G., & Scott, J. G. (2010). The horseshoe estimator for sparse signals. Biometrika, 97(2), 465-480.
3. Bishop, C. M. (2006). Pattern Recognition and Machine Learning. Springer.
 