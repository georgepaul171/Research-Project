# Adaptive Elastic Horseshoe (AEH) Prior Documentation

## Overview

The Adaptive Elastic Horseshoe (AEH) prior is a novel Bayesian prior that combines the strengths of elastic net regularization with the horseshoe prior's heavy-tailed properties. This prior is specifically designed for building energy performance analysis, where it provides adaptive feature selection and uncertainty quantification.

The AEH prior represents a significant advancement in Bayesian modeling for building energy analysis. Traditional priors often struggle with the complex, multi-scale nature of energy data, where features can exhibit both sparse and dense patterns simultaneously. The AEH prior addresses this challenge by introducing an adaptive mechanism that can seamlessly transition between different regularization regimes based on the underlying data structure.

## Mathematical Formulation

### Prior Structure

The AEH prior combines three key components:

1. **Elastic Net Component**:
   ```math
   \text{elastic\_penalty} = \alpha \|w\|_1 + (1-\alpha)\|w\|_2^2
   ```
   where:
   - $\alpha \in [0.1, 0.9]$ is the adaptive mixing parameter
   - $\|w\|_1$ is the L1 norm (lasso)
   - $\|w\|_2^2$ is the L2 norm (ridge)

The elastic net component provides a flexible regularization framework that can adapt to different feature importance patterns. The mixing parameter α dynamically adjusts the balance between L1 and L2 regularization, allowing the model to capture both sparse and dense solutions. This is particularly valuable in energy analysis, where some features may have strong individual effects (captured by L1) while others may contribute through complex interactions (captured by L2).

2. **Horseshoe Component**:
   ```math
   \text{horseshoe\_scale} = \frac{1}{\frac{w^2}{2\tau} + \beta \cdot \text{elastic\_penalty}}
   ```
   where:
   - $\tau$ is the global shrinkage parameter
   - $\beta$ is the adaptive regularization strength

The horseshoe component introduces heavy-tailed properties that are crucial for robust feature selection. Unlike traditional shrinkage priors, the horseshoe component allows for stronger shrinkage of irrelevant features while preserving the signal from important features. This is particularly important in energy analysis, where the signal-to-noise ratio can vary significantly across different features.

3. **Adaptive Updates**:
   ```math
   \text{momentum}_{t+1} = \rho \cdot \text{momentum}_t + \gamma \cdot \text{gradient}
   \lambda_{t+1} = \lambda_t + \text{momentum}_{t+1}
   ```
   where:
   - $\rho \in [0,1]$ is the momentum parameter
   - $\gamma$ is the learning rate
   - $\text{gradient}$ is the update direction

The adaptive update mechanism ensures stable optimization while allowing the prior to learn from the data. The momentum-based updates help prevent oscillations and improve convergence, while the adaptive learning rate ensures that the updates are appropriate for the current state of the optimization.

## Key Parameters

1. **Lambda ($\lambda$)**:
   - Type: Vector of local shrinkage parameters
   - Initialization: Ones vector
   - Update: Momentum-based adaptive updates
   - Purpose: Controls local feature shrinkage

2. **Tau ($\tau$)**:
   - Type: Scalar global shrinkage parameter
   - Initialization: 1.0
   - Update: Adaptive based on feature importance
   - Purpose: Controls global shrinkage strength

3. **Alpha ($\alpha$)**:
   - Type: Scalar mixing parameter
   - Range: [0.1, 0.9]
   - Initialization: 0.5
   - Update: Adaptive based on feature importance distribution
   - Purpose: Balances L1 and L2 regularization

4. **Beta ($\beta$)**:
   - Type: Scalar regularization strength
   - Range: [0.1, 10.0]
   - Initialization: 1.0
   - Update: Adaptive based on uncertainty
   - Purpose: Controls overall regularization strength

5. **Gamma ($\gamma$)**:
   - Type: Scalar learning rate
   - Value: 0.1
   - Purpose: Controls adaptation speed

6. **Rho ($\rho$)**:
   - Type: Scalar momentum parameter
   - Value: 0.9
   - Purpose: Controls momentum influence

## Update Mechanism

The update mechanism of the AEH prior is designed to be both robust and adaptive. It combines several sophisticated techniques to ensure stable learning while maintaining the flexibility needed for complex energy data.

### 1. Feature Importance and Uncertainty Calculation
```python
importance = np.array([np.clip(m[j]**2, 1e-10, None) for j in indices])
uncertainty = np.array([np.clip(np.diag(S)[j], 1e-10, None) for j in indices])
```

This step calculates two crucial quantities: feature importance and uncertainty. The importance measure captures the strength of each feature's contribution, while the uncertainty measure reflects our confidence in these contributions. These calculations are fundamental to the adaptive nature of the prior, as they guide the subsequent parameter updates.

### 2. Elastic Net Penalty Computation
```python
elastic_penalty = alpha * np.abs(importance) + (1 - alpha) * importance**2
```

### 3. Horseshoe Scaling
```python
horseshoe_scale = 1 / (importance / (2 * tau) + beta * elastic_penalty)
```

### 4. Momentum-Based Updates
```python
gradient = -horseshoe_scale + beta * elastic_penalty
momentum = rho * momentum + gamma * gradient
lambda_new = np.clip(lambda_old + momentum, 1e-10, None)
```

### 5. Adaptive Parameter Updates
```python
# Update alpha based on feature importance distribution
importance_ratio = np.mean(importance) / (np.std(importance) + 1e-10)
alpha_new = np.clip(alpha_old + gamma * (0.5 - importance_ratio), 0.1, 0.9)

# Update beta based on uncertainty
uncertainty_ratio = np.mean(uncertainty) / (np.std(uncertainty) + 1e-10)
beta_new = np.clip(beta_old + gamma * (1.0 - uncertainty_ratio), 0.1, 10.0)
```

## Advantages Over Existing Priors

The AEH prior offers several significant advantages over traditional priors, particularly in the context of building energy analysis. These advantages stem from its unique combination of elastic net properties, horseshoe characteristics, and adaptive learning mechanisms.

1. **Compared to Hierarchical Prior**:
   - More flexible shrinkage behavior
   - Adaptive parameter learning
   - Better handling of feature interactions

The hierarchical prior, while effective in many cases, can be too rigid in its shrinkage behavior. The AEH prior overcomes this limitation by introducing adaptive mechanisms that can adjust the shrinkage based on the data characteristics. This is particularly important in energy analysis, where the importance of features can vary significantly across different building types and operating conditions.

2. **Compared to Spike-Slab Prior**:
   - Continuous instead of binary shrinkage
   - More stable optimization
   - Better uncertainty quantification

3. **Compared to Horseshoe Prior**:
   - More stable hyperparameter sensitivity
   - Incorporates elastic net properties
   - Adaptive learning mechanism

## Implementation Details

The implementation of the AEH prior requires careful consideration of several factors to ensure both computational efficiency and numerical stability. The following sections detail the key implementation aspects and best practices.

### Initialization
```python
self.group_prior_hyperparams[group] = {
    'lambda': np.ones(len(indices)),
    'tau': 1.0,
    'alpha': 0.5,
    'beta': 1.0,
    'gamma': 0.1,
    'rho': 0.9,
    'momentum': np.zeros(len(indices))
}
```

The initialization strategy is crucial for the success of the AEH prior. The chosen values provide a balanced starting point that allows the prior to adapt to the data while maintaining stability. The momentum initialization to zero ensures a smooth start to the optimization process.

### Numerical Stability
- All updates include clipping to prevent numerical instability
- Small epsilon (1e-10) added to denominators
- Momentum-based updates for stability

### Convergence Properties
- The prior converges when:
  - Feature importance distribution stabilizes
  - Uncertainty estimates become consistent
  - Parameter updates become small

## Usage in Building Energy Analysis

The AEH prior is particularly well-suited for building energy analysis due to its ability to handle the complex, multi-scale nature of energy data. This section explains how the prior can be effectively applied in this domain.

### Energy Feature Group
The AEH prior is particularly effective for energy-related features because:
1. It can handle both sparse and dense solutions
2. It adapts to feature importance patterns
3. It provides robust uncertainty quantification
4. It maintains stability in optimization

The energy feature group often contains a mix of strongly influential features (like building size and HVAC system type) and more subtle features (like occupancy patterns and weather interactions). The AEH prior's ability to adapt to these different types of features makes it particularly valuable in this context.

### Feature Selection
The prior automatically:
1. Identifies important energy features
2. Adapts to feature interactions
3. Provides uncertainty-aware selection
4. Balances between sparse and dense solutions

## Performance Considerations

The performance of the AEH prior is influenced by several factors, including computational complexity, memory requirements, and optimization stability. Understanding these aspects is crucial for effective implementation.

### Computational Complexity
- Time complexity: O(n_features) per update
- Space complexity: O(n_features)
- Memory requirements: Moderate

The computational requirements of the AEH prior are reasonable for most practical applications. The linear complexity in both time and space makes it suitable for datasets of various sizes, while the moderate memory requirements ensure it can be implemented on standard hardware.

### Optimization Stability
- Momentum-based updates prevent oscillations
- Adaptive learning rates maintain stability
- Clipping operations prevent numerical issues

## Best Practices

Implementing the AEH prior effectively requires attention to several key aspects. The following guidelines help ensure optimal performance and reliable results.

1. **Parameter Tuning**:
   - Start with default values
   - Monitor convergence
   - Adjust gamma if needed
   - Consider feature group sizes

The parameter tuning process should be approached systematically, starting with the default values and making adjustments based on observed behavior. Regular monitoring of convergence and performance metrics is essential for identifying when adjustments are needed.

2. **Monitoring**:
   - Track parameter evolution
   - Monitor feature importance
   - Check uncertainty estimates
   - Verify convergence

3. **Troubleshooting**:
   - If unstable: decrease gamma
   - If slow convergence: increase gamma
   - If over-shrinkage: adjust beta range
   - If under-shrinkage: adjust alpha range

## Future Improvements

The AEH prior, while already powerful, has several potential areas for enhancement. These improvements could further increase its effectiveness in building energy analysis and other applications.

1. **Potential Enhancements**:
   - Group-specific adaptation rates
   - Hierarchical structure for parameters
   - More sophisticated momentum schemes
   - Uncertainty-aware adaptation

These potential improvements would build upon the current strengths of the AEH prior while addressing specific challenges in building energy analysis. The development of group-specific adaptation rates, for example, could better handle the different characteristics of various feature groups in energy data.

## References

1. Original Horseshoe Prior:
   - Carvalho, C. M., Polson, N. G., & Scott, J. G. (2010)
   - "The horseshoe estimator for sparse signals"

2. Elastic Net:
   - Zou, H., & Hastie, T. (2005)
   - "Regularization and variable selection via the elastic net"

3. Adaptive Learning:
   - Kingma, D. P., & Ba, J. (2014)
   - "Adam: A method for stochastic optimization" 