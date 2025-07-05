# Model Architecture: Adaptive Bayesian Regression Models

## Mathematical Framework

### General Bayesian Linear Model
The research employs a hierarchical Bayesian linear regression framework:

```
y_i ~ Normal(μ_i, σ²)                    # Likelihood
μ_i = X_i^T β                            # Linear predictor
β_j ~ Prior(θ_j)                         # Prior on coefficients
θ_j ~ Hyperprior(φ)                      # Hyperprior on prior parameters
σ² ~ InverseGamma(α_σ, β_σ)             # Prior on noise variance
```

Where:
- `y_i` is the site energy use intensity for building i
- `X_i` is the feature vector for building i
- `β` is the coefficient vector
- `σ²` is the noise variance
- `θ_j` are prior parameters for each coefficient
- `φ` are hyperprior parameters

## Prior Specifications

### 1. Adaptive Elastic Horseshoe (AEH) Prior

The AEH prior combines the benefits of the horseshoe prior with adaptive shrinkage:

```
β_j ~ Normal(0, τ²λ_j²)                  # Coefficient prior
λ_j ~ Half-Cauchy(0, 1)                  # Local shrinkage
τ ~ Half-Cauchy(0, τ_0)                  # Global shrinkage
τ_0 ~ Gamma(a, b)                        # Hyperprior on global shrinkage
```

#### Mathematical Properties
- **Sparsity**: Induces exact zeros through strong shrinkage
- **Adaptivity**: Local shrinkage parameters adapt to data
- **Heavy Tails**: Allows large coefficients when supported by data
- **Scale Invariance**: Automatically adapts to feature scales

#### Implementation Details
```python
def aeh_prior_log_prob(beta, lambda_j, tau, tau_0):
    """Log probability of AEH prior"""
    # Global shrinkage
    log_prob = stats.halfcauchy.logpdf(tau, 0, tau_0)
    
    # Local shrinkage for each coefficient
    for j in range(len(beta)):
        log_prob += stats.halfcauchy.logpdf(lambda_j[j], 0, 1)
        log_prob += stats.norm.logpdf(beta[j], 0, tau * lambda_j[j])
    
    return log_prob
```

### 2. Hierarchical Prior

The hierarchical prior allows different groups of features to have different prior specifications:

```
β_j ~ Normal(0, σ_j²)                    # Coefficient prior
σ_j² ~ InverseGamma(α_j, β_j)           # Group-specific variance
```

#### Group Definitions
- **Energy Features**: electric_eui, fuel_eui, energy_mix, energy_intensity_ratio
- **Building Features**: floor_area_log, building_age_log, energy_star_rating_normalized
- **Environmental Features**: ghg_emissions_int_log, ghg_per_area
- **Interaction Features**: All squared terms and interactions

#### Implementation
```python
group_prior_types = {
    'energy': 'adaptive_elastic_horseshoe',
    'building': 'hierarchical',
    'interaction': 'spike_slab'
}
```

### 3. Spike-Slab Prior

The spike-slab prior is a mixture of a point mass at zero and a normal distribution:

```
β_j ~ (1 - π_j) δ_0 + π_j Normal(0, σ_j²)
π_j ~ Beta(a_j, b_j)                     # Inclusion probability
σ_j² ~ InverseGamma(α_j, β_j)           # Variance when included
```

Where `δ_0` is a point mass at zero.

## Model Implementation

### 1. AdaptivePriorARD Class

The main model class implements the adaptive prior framework:

```python
@dataclass
class AdaptivePriorConfig:
    alpha_0: float = 1e-6                # Initial noise precision
    beta_0: float = 1e-6                 # Initial weight precision
    max_iter: int = 100                  # Maximum EM iterations
    tol: float = 1e-4                    # Convergence tolerance
    prior_type: str = 'hierarchical'     # Prior specification
    adaptation_rate: float = 0.1         # Prior adaptation rate
    use_hmc: bool = True                 # Use Hamiltonian Monte Carlo
    uncertainty_calibration: bool = True # Apply uncertainty calibration
```

### 2. Expectation-Maximization Algorithm

The model uses an EM algorithm for parameter estimation:

#### E-Step: Posterior Sampling
```python
def e_step(self, X, y, current_params):
    """Expectation step: sample from posterior"""
    # Use Hamiltonian Monte Carlo for posterior sampling
    if self.config.use_hmc:
        samples = self.hmc_sampling(X, y, current_params)
    else:
        samples = self.variational_inference(X, y, current_params)
    
    return samples
```

#### M-Step: Parameter Updates
```python
def m_step(self, X, y, posterior_samples):
    """Maximization step: update parameters"""
    # Update prior parameters based on posterior samples
    updated_params = self.update_priors(posterior_samples)
    
    # Update noise variance
    noise_var = self.update_noise_variance(X, y, posterior_samples)
    
    return updated_params, noise_var
```

### 3. Hamiltonian Monte Carlo Implementation

For posterior sampling, the model implements HMC:

```python
def hmc_sampling(self, X, y, params, n_steps=200, epsilon=0.001):
    """Hamiltonian Monte Carlo sampling"""
    def potential_energy(beta):
        """Negative log posterior (potential energy)"""
        log_likelihood = self.log_likelihood(X, y, beta)
        log_prior = self.log_prior(beta, params)
        return -(log_likelihood + log_prior)
    
    def gradient_energy(beta):
        """Gradient of potential energy"""
        return -self.gradient_log_posterior(X, y, beta, params)
    
    # HMC sampling implementation
    samples = []
    current_beta = np.random.normal(0, 1, X.shape[1])
    
    for step in range(n_steps):
        # Momentum update
        momentum = np.random.normal(0, 1, len(current_beta))
        
        # Leapfrog integration
        for leapfrog_step in range(self.config.hmc_leapfrog_steps):
            momentum -= 0.5 * epsilon * gradient_energy(current_beta)
            current_beta += epsilon * momentum
            momentum -= 0.5 * epsilon * gradient_energy(current_beta)
        
        # Metropolis acceptance
        current_energy = potential_energy(current_beta)
        proposed_energy = potential_energy(current_beta)
        
        if np.random.random() < np.exp(current_energy - proposed_energy):
            samples.append(current_beta.copy())
    
    return np.array(samples)
```

## Uncertainty Quantification

### 1. Predictive Uncertainty

The model provides predictive uncertainty through posterior predictive sampling:

```python
def predict_with_uncertainty(self, X_new):
    """Predict with uncertainty quantification"""
    predictions = []
    
    # Sample from posterior
    posterior_samples = self.get_posterior_samples()
    
    for sample in posterior_samples:
        # Make prediction with current sample
        pred = X_new @ sample
        predictions.append(pred)
    
    predictions = np.array(predictions)
    
    # Compute statistics
    mean_pred = np.mean(predictions, axis=0)
    std_pred = np.std(predictions, axis=0)
    
    return mean_pred, std_pred
```

### 2. Uncertainty Calibration

Post-hoc uncertainty calibration is applied to improve reliability:

```python
def calibrate_uncertainty(self, y_true, y_pred, y_std):
    """Calibrate uncertainty estimates"""
    # Compute empirical coverage
    coverages = [0.5, 0.8, 0.9, 0.95, 0.99]
    empirical_coverages = []
    
    for coverage in coverages:
        z_score = stats.norm.ppf(1 - (1 - coverage) / 2)
        lower = y_pred - z_score * y_std
        upper = y_pred + z_score * y_std
        empirical = np.mean((y_true >= lower) & (y_true <= upper))
        empirical_coverages.append(empirical)
    
    # Find calibration factor
    calibration_factor = np.mean(np.array(coverages) / np.array(empirical_coverages))
    
    return calibration_factor
```

## Model Comparison Framework

### 1. Baseline Models

#### Linear Regression
```
y_i ~ Normal(X_i^T β, σ²)
β_j ~ Uniform(-∞, ∞)  # Uninformative prior
σ² ~ InverseGamma(ε, ε)  # Jeffreys prior
```

#### Bayesian Ridge
```
y_i ~ Normal(X_i^T β, σ²)
β_j ~ Normal(0, τ²)   # Ridge prior
τ² ~ InverseGamma(α, β)  # Hyperprior
σ² ~ InverseGamma(α, β)  # Noise prior
```

### 2. Performance Metrics

#### Predictive Accuracy
- **RMSE**: Root Mean Square Error
- **MAE**: Mean Absolute Error  
- **R²**: Coefficient of Determination

#### Uncertainty Quality
- **Calibration Error**: Difference between nominal and empirical coverage
- **Interval Width**: Average width of prediction intervals
- **Coverage Probability**: Proportion of true values within intervals

#### Model Interpretability
- **Feature Importance**: Standardized coefficient magnitudes
- **Sparsity**: Number of effectively zero coefficients
- **SHAP Values**: Local and global feature contributions

## Computational Considerations

### 1. Scalability
- **Sample Size**: Models scale to thousands of buildings
- **Feature Count**: Efficient with dozens of features
- **Computational Cost**: HMC sampling is the main bottleneck

### 2. Convergence Diagnostics
- **Trace Plots**: Monitor parameter convergence
- **Gelman-Rubin Statistic**: Assess mixing and convergence
- **Effective Sample Size**: Measure sampling efficiency

### 3. Numerical Stability
- **Feature Scaling**: StandardScaler applied to all features
- **Prior Specification**: Careful choice of hyperparameters
- **Regularization**: Prevents overfitting and numerical issues

## Model Validation

### 1. Cross-Validation
- **K-Fold CV**: K=3 for computational efficiency
- **Stratified Sampling**: Maintains target distribution
- **Repeated CV**: Multiple runs for stability

### 2. Posterior Predictive Checks
- **Residual Analysis**: Check model adequacy
- **Coverage Assessment**: Validate uncertainty estimates
- **Out-of-Sample Performance**: Test generalization

### 3. Sensitivity Analysis
- **Prior Sensitivity**: Test different prior specifications
- **Hyperparameter Sensitivity**: Vary key parameters
- **Data Sensitivity**: Robustness to data changes

---

*This model architecture document provides the mathematical and computational foundation for the Bayesian models used in the research. It should be referenced when implementing, interpreting, or extending the models.* 