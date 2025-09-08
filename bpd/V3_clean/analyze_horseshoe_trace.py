import arviz as az
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os

# Path setup
results_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results_pymc_horseshoe_comprehensive')
trace_path = os.path.join(results_dir, 'trace.nc')

feature_names = [
    "ghg_emissions_int_log",
    "floor_area_log",
    "electric_eui",
    "fuel_eui",
    "energy_star_rating_normalized",
    "energy_mix",
    "building_age_log",
    "floor_area_squared",
    "energy_intensity_ratio",
    "building_age_squared",
    "energy_star_rating_squared",
    "ghg_per_area"
]

# 1. Load the trace
trace = az.from_netcdf(trace_path)

# 2. Posterior summary table
summary = az.summary(trace, round_to=4)
summary.to_csv(os.path.join(results_dir, 'trace_posterior_summary.csv'))
print("Posterior summary saved.")

# 3. Trace plots for all key parameters
az.plot_trace(trace, var_names=['intercept', 'sigma', 'tau'])
plt.tight_layout()
plt.savefig(os.path.join(results_dir, 'trace_plots_main.png'), dpi=300, bbox_inches='tight')
plt.close()

# Trace plots for all coefficients
az.plot_trace(trace, var_names=['coeffs'])
plt.tight_layout()
plt.savefig(os.path.join(results_dir, 'trace_plots_coeffs.png'), dpi=300, bbox_inches='tight')
plt.close()

# 4. R-hat and ESS diagnostics
rhat = az.rhat(trace)
ess = az.ess(trace)

rhat_vals = []
for var in rhat.data_vars:
    vals = rhat[var].values.flatten()
    rhat_vals.extend(vals)
rhat_vals = np.array(rhat_vals)
ess_vals = []
for var in ess.data_vars:
    vals = ess[var].values.flatten()
    ess_vals.extend(vals)
ess_vals = np.array(ess_vals)
with open(os.path.join(results_dir, 'diagnostics_from_trace.txt'), 'w') as f:
    f.write(f"R-hat (max): {np.max(rhat_vals):.3f}\n")
    f.write(f"R-hat (mean): {np.mean(rhat_vals):.3f}\n")
    f.write(f"ESS (min): {np.min(ess_vals):.1f}\n")
    f.write(f"ESS (mean): {np.mean(ess_vals):.1f}\n")
    f.write(f"ESS (sum): {int(np.sum(ess_vals))}\n")
print("Diagnostics saved.")

# 5. Posterior distributions for coefficients

print('coeffs coords:', trace.posterior['coeffs'].coords)
coeff_params = [name for name in summary.index if 'coeffs' in name][:len(feature_names)]
for idx, param in enumerate(coeff_params):
    try:
        print(f'Plotting posterior for coeffs_dim_0={int(idx)}')
        az.plot_posterior(trace, var_names=['coeffs'], coords={'coeffs_dim_0': [int(idx)]})
        plt.title(f'Posterior of {feature_names[idx]} (coeffs[{idx}])')
        plt.tight_layout()
        plt.savefig(os.path.join(results_dir, f'posterior_{feature_names[idx]}.png'), dpi=300, bbox_inches='tight')
        plt.close()
    except KeyError as e:
        print(f"Skipping idx={idx} due to KeyError: {e}")

# 6. Feature importance plot
coeff_means = summary.loc[coeff_params, 'mean'].values[:len(feature_names)]
coeff_stds = summary.loc[coeff_params, 'sd'].values[:len(feature_names)]
importance_order = np.argsort(np.abs(coeff_means))[::-1]
fig, ax = plt.subplots(figsize=(12, 8))
y_pos = np.arange(len(feature_names))
ax.barh(y_pos, coeff_means[importance_order], xerr=coeff_stds[importance_order], capsize=5, alpha=0.7)
ax.set_yticks(y_pos)
ax.set_yticklabels([feature_names[i] for i in importance_order])
ax.set_xlabel('Coefficient Value')
ax.set_title('Horseshoe: Feature Importance (Posterior Mean ± Std)')
ax.axvline(x=0, color='black', linestyle='-', alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(results_dir, 'feature_importance_from_trace.png'), dpi=300, bbox_inches='tight')
plt.close()

print("All analyses from trace complete.") 