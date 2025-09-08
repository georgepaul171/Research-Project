# Research Journey: Adaptive Elastic Horseshoe (AEH) Prior for Bayesian Regression

## 1. Introduction & Motivation

The goal of this project was to implement and evaluate a novel **Adaptive Elastic Horseshoe (AEH) prior** for Bayesian linear regression, with a focus on building energy data. The AEH prior was designed to combine the strong shrinkage of the horseshoe prior with adaptive, group-wise flexibility, aiming to improve feature selection and uncertainty quantification compared to standard Bayesian approaches.

## 2. Implementation

- Developed a custom Bayesian regression model with support for group priors (energy, building, interaction features).
- Implemented the AEH prior for the energy group, with hierarchical or horseshoe priors for other groups.
- Added robust diagnostics: prediction ranges, feature weights, trace plots, and hyperparameter logging.

## 3. Early Results & Problems

- **Initial runs showed severe underfitting:**
  - The model predicted values in a very narrow range (e.g., 16–24), while the true target spanned 4–154.
  - Diagnostics revealed that the model was not using the full information in the data.
- **LinearRegression and BayesianRidge baselines** could predict the full range, confirming the data and features were informative.

## 4. Debugging & Iterative Improvements

- **Relaxed the prior:**
  - Reduced `beta_0` (prior strength) and disabled group sparsity/dynamic shrinkage.
  - Increased EM iterations and tightened convergence tolerance.
- **Added extensive logging:**
  - Saved weights, prediction ranges, and hyperparameters at each EM iteration.
  - Logged AEH-specific parameters (lambda, tau, etc.) to check for numerical stability.
- **Tested HMC and EM:**
  - HMC acceptance probabilities were always zero with AEH, indicating a pathological posterior.
  - Even with EM only, the model was still conservative, but more stable.

## 5. Diagnostics & Comparison

- **Trace plots:**
  - For AEH, traces were flat (no mixing), confirming the posterior was too sharp for HMC.
  - For BayesianRidge (using bootstrapped fits), traces were wiggly, showing good mixing and a well-behaved posterior.
- **Weights:**
  - AEH model weights were all moderate (e.g., -0.65 to 0.51), never large.
  - BayesianRidge weights were much larger (e.g., 44.7, 20.9, -18.2), allowing the model to fit the full range.
- **Prediction ranges:**
  - AEH: 2.5–77.1 (never the full range, even with minimal shrinkage).
  - BayesianRidge: -26.6–153.8 (matches the true range).

## 6. Research Insights & Implications

- **The AEH prior is numerically stable after refactoring, but extremely conservative in practice.**
- **Even with the weakest possible prior, the AEH model keeps weights small and cannot fit the full range of the data.**
- **BayesianRidge and minimal Bayes models, with global weak priors, allow weights to become large and fit the data well.**
- **HMC cannot explore the AEH posterior (acceptance probability ≈ 0, flat traces), while it mixes well for simpler priors.**

### **Key Takeaways:**
- Complex priors like AEH can be hard to tune and may overly constrain the model, limiting both predictive performance and posterior exploration.
- Simpler priors (BayesianRidge) are more flexible and easier to use in practice.
- Negative results are valuable: showing when a new prior is *too* strong is an important contribution to the literature.
- Empirical diagnostics (trace plots, weights, prediction ranges) are essential for understanding model behavior.

## 7. Conclusion

This research journey demonstrates the importance of:
- Careful prior design and empirical validation in Bayesian modeling.
- Honest reporting of both successes and limitations.
- Providing reproducible diagnostics and comparisons for future researchers.

**The AEH prior, while theoretically appealing, proved too restrictive for this application. This is a meaningful and publishable result, and the code and diagnostics developed here will be valuable for others exploring advanced Bayesian priors.** 