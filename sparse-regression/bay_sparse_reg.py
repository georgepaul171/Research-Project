# ---
# jupytext:
#   formats: ipynb,py:light
#   jupytext_format_version: '1.3'
#   kernelspec:
#     name: python3
#     display_name: Python 3 (ipykernel)
#     language: python
# ---

# %% [markdown]
# # Improved Bayesian Sparse Regression with Horseshoe Prior (Non‑Centered)
#
# - Predictors standardized to mean 0 and unit variance  
# - Non‑centered horseshoe prior with lighter-tail normals  
# - NUTS sampler with higher target acceptance to reduce divergences

# %%
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

import jax.numpy as jnp
from jax import random

import numpyro
import numpyro.distributions as dist
from numpyro.infer import MCMC, NUTS, Predictive

import arviz as az
import matplotlib.pyplot as plt

# %% [markdown]
# ## 1. Load and Preprocess Data

# %%
df = pd.read_csv('Data/Data_Ready_to_Model.csv')

def floor_band(n):
    if n < 994:
        return '1-9'
    elif n == 994:
        return '10-14'
    elif n == 995:
        return '15+'
    else:
        return np.nan

df['NFLOOR_CAT'] = df['NFLOOR'].apply(floor_band).astype('category')

required = [
    'EUI_kWh_per_sqmt', 'SQMT', 'FLCEILHT', 'MONUSE', 'OCCUPYP',
    'WKHRS', 'NWKER', 'HEATP', 'COOLP', 'DAYLTP', 'HDD65', 'CDD65', 'NFLOOR_CAT'
]
df_model = df.dropna(subset=required).copy()

dummies = pd.get_dummies(df_model['NFLOOR_CAT'], prefix='NFLOOR_CAT').iloc[:, 1:]
df_ohe = pd.concat([df_model.reset_index(drop=True), dummies], axis=1)

numeric_cols = [
    'SQMT', 'FLCEILHT', 'MONUSE', 'OCCUPYP',
    'WKHRS', 'NWKER', 'HEATP', 'COOLP',
    'DAYLTP', 'HDD65', 'CDD65'
]
dummy_cols = [c for c in df_ohe.columns if c.startswith('NFLOOR_CAT_')]

X = df_ohe[numeric_cols + dummy_cols].values
y = df_ohe['EUI_kWh_per_sqmt'].values

scaler = StandardScaler()
X_std = scaler.fit_transform(X)

X_jax = jnp.array(X_std)
y_jax = jnp.array(y)

# %% [markdown]
# ## 2. Define Non‑Centered Horseshoe Model

# %%
def model(X, y=None):
    n, p = X.shape
    tau = numpyro.sample('tau', dist.HalfNormal(1.0))
    lam = numpyro.sample('lam', dist.HalfNormal(1.0).expand([p]))
    eta = numpyro.sample('eta', dist.Normal(0, 1).expand([p]))
    beta = tau * lam * eta
    numpyro.deterministic('beta', beta)

    b0 = numpyro.sample('b0', dist.Normal(0, 10.0))
    sigma = numpyro.sample('sigma', dist.HalfNormal(5.0))

    mu = jnp.dot(X, beta) + b0
    numpyro.sample('obs', dist.Normal(mu, sigma), obs=y)

# %% [markdown]
# ## 3. Run MCMC Sampling

# %%
rng_key = random.PRNGKey(0)
# Set target_accept_prob in NUTS, not in MCMC
kernel = NUTS(model, max_tree_depth=12, target_accept_prob=0.9)
mcmc = MCMC(kernel, num_warmup=1500, num_samples=1500, num_chains=2)
mcmc.run(rng_key, X=X_jax, y=y_jax)
mcmc.print_summary()

# %% [markdown]
# ## 4. Posterior Analysis

# %%
idata = az.from_numpyro(mcmc)
az.plot_forest(idata, var_names=['beta'], combined=True)
plt.title('Posterior Distributions of Sparse Coefficients')
plt.show()

# %% [markdown]
# ## 5. Posterior Predictive & Model Performance

# %%
predictive = Predictive(model, posterior_samples=mcmc.get_samples())
y_rep = predictive(rng_key, X=X_jax)['obs']
y_pred = np.mean(np.array(y_rep), axis=0)

print(f"RMSE: {np.sqrt(mean_squared_error(y, y_pred)):.2f}")
print(f"MAE:  {mean_absolute_error(y, y_pred):.2f}")
print(f"R²:   {r2_score(y, y_pred):.3f}")

# %% [markdown]
# ## Summary of Results and Next Steps
#
# **Posterior Distributions of Sparse Coefficients**  
# - **β[0] (SQMT)**: Strongly negative (95% CI ≈ [–22, –2]), indicating larger buildings have lower EUI per m² (economies of scale).  
# - **β[4] (OCCUPYP) & β[5] (WKHRS)**: Clear positive effects (~20–30), showing that higher occupancy density and longer operating hours drive up energy use.  
# - **β[6] (NWKER)**: Moderately positive (~15, CI ≈ [5, 25]), more employees → more energy per m².  
# - **Many coefficients** (e.g. indices 2, 3, 8, 10–12) have 95% CIs overlapping zero—shrunk toward zero by the horseshoe prior.  
# - **Floor‑band dummies**: The 15+ floors category hints at a positive effect but with a wider CI that sometimes crosses zero, suggesting more data needed for certainty.
#
# **Model Performance**  
# - **RMSE:** ~107 kWh/m²  
# - **MAE:** ~ 73.5 kWh/m²  
# - **R²:** ~ 0.14  
#
# Although we’ve identified the strongest predictors and pruned irrelevant ones, the model currently explains only ~14% of EUI variance.  
#
# **Next Steps**  
# 1. Incorporate a **hierarchical** structure (e.g., by climate zone or year‐built) to share information across subgroups.  
# 2. Add **interaction terms** (e.g., SQMT × OCCUPYP, HDD65 × COOLP) to capture conditional effects.  
# 3. Bring in **additional covariates** (detailed HVAC controls, retrofit status, IoT/smart‐meter time‐series) to boost predictive power.  
# 4. Tune priors or sampler settings further if any residual divergences reappear.
# %%
