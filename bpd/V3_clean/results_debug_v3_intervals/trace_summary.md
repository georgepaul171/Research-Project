# Trace Diagnostics Summary

## BayesianRidge (MinimalBayes)
- Trace plots (trace_minimal_bayes_*.png) show wiggly, well-mixing traces for top features.
- This indicates the posterior is well-behaved and HMC (or bootstrapping) can explore it.

## AEH Prior Model
- Trace plots for AEH (trace_*.png) are flat or nearly flat.
- This means the posterior is too sharp or the prior is too strong, so HMC cannot explore.

## Research Implication
- Complex priors like AEH can be numerically stable but may overly constrain the model, limiting posterior exploration.
- Simpler priors (BayesianRidge) allow for better mixing and more flexible fits.
- This highlights the importance of empirical diagnostics and careful prior design in Bayesian modeling.
