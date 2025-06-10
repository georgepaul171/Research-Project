# Mathematical Documentation of Adaptive Elastic Horseshoe (AEH) Prior

## Why These Properties Matter in the Energy Domain

Building energy datasets are characterised by high heterogeneity, complex feature interactions, and varying levels of sparsity and density among predictors. In this domain:

- **Feature Importance is Highly Variable:** Some features (e.g., building size, HVAC type) have strong, direct effects, while others (e.g., occupancy, weather) may have subtle or context-dependent impacts. A prior must adaptively balance between selecting a few strong predictors and allowing for dense, distributed effects.

- **Sparsity and Density Coexist:** Energy data often contains both sparse signals (few important features) and dense signals (many small but relevant features). Traditional priors may over-shrink or under-shrink, missing important patterns.

- **Uncertainty Quantification:** Decisions in energy management and policy require not just point estimates, but also reliable measures of uncertainty. The prior must provide robust uncertainty quantification, especially when data is noisy or incomplete.

- **Multi-Scale and Correlated Features:** Features can be correlated and operate at different scales (e.g., hourly weather vs. annual building characteristics). The prior must be stable and flexible enough to handle these complexities.

The AEH prior addresses these needs by:
- **Adaptive Regularition:** Balancing L1 (sparse) and L2 (dense) penalties to match the underlying data structure.
- **Heavy-Tailed Shrinkage:** Allowing large coefficients for important features while strongly shrinking irrelevant ones, thanks to the horseshoe component.
- **Momentum-Based Updates:** Ensuring stable and efficient learning, even in the presence of noisy or highly variable data.
- **Robust Uncertainty Quantification:** Providing reliable estimates of both feature importance and uncertainty.

These properties make the AEH prior especially well-suited for the challenges of building energy analysis, enabling more accurate, interpretable, and actionable models.

## Automatic Relevance Determination (ARD) in the AEH Prior

The AEH prior incorporates the principle of Automatic Relevance Determination (ARD), a Bayesian approach that allows the model to automatically infer the importance of each feature. In classical ARD, each feature is assigned its own relevance (shrinkage) parameter, so that unimportant features are strongly shrunk towards zero while important features are retained.

In the AEH prior, this is achieved through the use of local shrinkage parameters `lambda_i` for each feature weight `w_i`. These parameters are updated adaptively during training:

- Each `lambda_i` acts as a feature-specific regularisation strength.
- If a feature is unimportant, its `lambda_i` will decrease, leading to stronger shrinkage of `w_i` towards zero.
- If a feature is important, its `lambda_i` will remain larger, allowing `w_i` to stay nonzero.
- The update rule `lambda_{t+1} = lambda_t + momentum_{t+1}` ensures that each feature's relevance is learned from the data.

This ARD-like mechanism enables the AEH prior to perform **automatic feature selection and relevance weighting**, making it especially effective for high-dimensional and heterogeneous datasets such as those found in the energy domain.

## 1. Mathematical Formulation

### 1.1 Prior Structure

The AEH prior combines three key components:

1. **Elastic Net Component**:

   `elastic_penalty(w) = alpha * ||w||_1 + (1 - alpha) * ||w||_2^2`
   
   where:
   - `w` is a vector of parameters in R^p
   - `alpha` is the adaptive mixing parameter in [0.1, 0.9]
   - `||w||_1 = sum_{i=1}^p |w_i|` is the L1 norm
   - `||w||_2^2 = sum_{i=1}^p w_i^2` is the L2 norm

2. **Horseshoe Component**:

   `horseshoe_scale(w) = 1 / (w^2 / (2 * tau) + beta * elastic_penalty(w))`
   
   where:
   - `tau > 0` is the global shrinkage parameter
   - `beta > 0` is the adaptive regularisation strength

3. **Combined Prior**:

   `p(w | alpha, beta, tau, lambda) ∝ prod_{i=1}^p [1 / sqrt(2 * pi * lambda_i)] * exp(-w_i^2 / (2 * lambda_i)) * horseshoe_scale(w)`
   
   where `lambda_i > 0` are local shrinkage parameters.

### 1.2 Adaptive Update Mechanism

The update mechanism follows a momentum-based approach:

1. **Momentum Update**:

   `momentum_{t+1} = rho * momentum_t + gamma * grad_w log p(w_t)`
   
   where:
   - `rho` is the momentum parameter in [0, 1]
   - `gamma` is the learning rate
   - `grad_w log p(w_t)` is the gradient of the log-posterior

2. **Parameter Updates**:

   ```
   lambda_{t+1} = lambda_t + momentum_{t+1}
   alpha_{t+1} = clip(alpha_t + gamma * (0.5 - importance_ratio), 0.1, 0.9)
   beta_{t+1} = clip(beta_t + gamma * (1.0 - uncertainty_ratio), 0.1, 10.0)
   ```

## 2. Theoretical Properties

### 2.1 Shrinkage Properties

The AEH prior exhibits the following shrinkage properties:

1. **Heavy-tailed Behavior**:

   `lim_{w -> ∞} [p(w) / p_normal(w)] = ∞`
   
   This ensures that the prior can accommodate large parameter values when supported by the data.

2. **Adaptive Shrinkage**:

   `shrinkage(w_i) = 1 / (1 + w_i^2 / (2 * tau * lambda_i) + beta * elastic_penalty(w_i))`
   
   This provides adaptive shrinkage that depends on both the parameter value and the elastic net penalty.

### 2.2 Convergence Properties

The AEH prior's update mechanism satisfies the following properties:

1. **Bounded Updates**:

   `||momentum_{t+1}|| <= gamma / (1 - rho) * ||grad_w log p(w_t)||`
   
   This ensures that updates remain bounded and stable.

2. **Convergence Conditions**:

   The algorithm converges when:

   ```
   ||momentum_{t+1}|| < epsilon_1
   ||lambda_{t+1} - lambda_t|| < epsilon_2
   ||alpha_{t+1} - alpha_t|| < epsilon_3
   ||beta_{t+1} - beta_t|| < epsilon_4
   ```

   for small positive constants `epsilon_1, epsilon_2, epsilon_3, epsilon_4`.

## 3. Implementation Details

### 3.1 Numerical Stability

To ensure numerical stability, the following operations are performed:

1. **Clipping Operations**:

   `clip(x, a, b) = max(a, min(b, x))`

2. **Small Constant Addition**:

   `stable_div(a, b) = a / (b + epsilon)`

   where `epsilon = 1e-10` is a small constant.

### 3.2 Computational Complexity

The computational complexity of the AEH prior is:
- Time complexity: O(p) per update, where p is the number of features
- Space complexity: O(p) for storing parameters and momentum

## 4. Proofs

### 4.1 Bounded Updates Proof

**Theorem:** The momentum updates in the AEH prior are bounded.

**Proof:**

```
||momentum_{t+1}|| = ||rho * momentum_t + gamma * grad_w log p(w_t)||
                  <= rho * ||momentum_t|| + gamma * ||grad_w log p(w_t)||
                  <= rho * (gamma / (1 - rho)) * ||grad_w log p(w_{t-1})|| + gamma * ||grad_w log p(w_t)||
                  <= gamma / (1 - rho) * ||grad_w log p(w_t)||
```

### 4.2 Convergence Proof

**Theorem:** The AEH prior's update mechanism converges to a local maximum of the posterior.

**Proof:**
1. The momentum updates are bounded (from previous proof)
2. The parameter updates are bounded by the clipping operations
3. The objective function is continuous and differentiable
4. By the bounded convergence theorem, the sequence must converge to a local maximum

## 5. References

1. Carvalho, C. M., Polson, N. G., & Scott, J. G. (2010). "The horseshoe estimator for sparse signals"
2. Zou, H., & Hastie, T. (2005). "Regularization and variable selection via the elastic net"
3. Kingma, D. P., & Ba, J. (2014). "Adam: A method for stochastic optimization"