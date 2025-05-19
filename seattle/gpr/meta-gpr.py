import pymc as pm
import pytensor.tensor as pt
import pandas as pd
import numpy as np
import arviz as az
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error
import warnings

warnings.filterwarnings("ignore", category=FutureWarning)

# Define columns
numeric_columns = [
    "YearBuilt",
    "NumberofFloors",
    "NumberofBuildings",
    "PropertyGFATotal",
    "ENERGYSTARScore",
    "Electricity(kWh)",
    "NaturalGas(kBtu)",
    "SteamUse(kBtu)",
    "GHGEmissionsIntensity",
    "SiteEUI(kBtu/sf)",
]

target_col = "SiteEUI(kBtu/sf)"

# Load data (assuming it's already cleaned and imputed)
df = pd.read_csv(
    "/Users/georgepaul/Desktop/Research-Project/seattle/data/seattle-data-cleaned.csv"
)[numeric_columns]

# Split into features/target
X = df.drop(columns=[target_col])
y = df[target_col]

# Train/test split (for meta-training and meta-testing)
X_meta_train, X_meta_test, y_meta_train, y_meta_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)  # Further split for meta-learning

# Function to create and train a GP model
def create_and_train_gp(X_train, y_train, X_test, initial_lengthscale, initial_variance):
    with pm.Model() as model:
        X_data = pm.Data("X_data", X_train)
        y_data = pm.Data("y_data", y_train)

        # Hyperparameters (now with potential initial values)
        lengthscale = pm.Gamma(
            "lengthscale", alpha=2, beta=1, testval=initial_lengthscale
        )  # Using testval for initialization
        variance = pm.HalfNormal(
            "variance", sigma=1, testval=initial_variance
        )  # Using testval
        noise_sigma = pm.HalfNormal("noise_sigma", sigma=1)

        cov_func = variance**2 * pm.gp.cov.ExpQuad(
            input_dim=X_train.shape[1], ls=lengthscale
        )
        gp = pm.gp.Marginal(cov_func=cov_func)
        y_ = gp.marginal_likelihood("y", X=X_data, y=y_data, noise=noise_sigma)

        trace = pm.sample(
            1000, tune=1000, target_accept=0.9, chains=1, cores=1, random_seed=42
        )

        # Prediction
        pm.Data.set("X_data", X_test)
        f_pred = gp.conditional("f_pred", X_test)
        posterior_pred = pm.sample_posterior_predictive(
            trace, var_names=["f_pred"], predictions=True, random_seed=42
        )

    f_samples = posterior_pred.predictions["f_pred"].values
    f_samples = f_samples.reshape(-1, f_samples.shape[-1])
    mu_pred = f_samples.mean(axis=0)
    std_pred = f_samples.std(axis=0)

    r2 = r2_score(y_test, mu_pred)
    rmse = np.sqrt(mean_squared_error(y_test, mu_pred))

    return {"r2": r2, "rmse": rmse, "trace": trace, "lengthscale": lengthscale, "variance": variance}


# Meta-training loop (simplified)
num_meta_train_tasks = 5  # Number of subsets to train on
meta_learning_rate = 0.1
initial_lengthscale = 1.0  # Initial guess for lengthscale
initial_variance = 1.0  # Initial guess for variance

for i in range(num_meta_train_tasks):
    # Sample a subset of the meta-train data (a "task")
    X_train_task, X_val_task, y_train_task, y_val_task = train_test_split(
        X_meta_train, y_meta_train, test_size=0.2
    )

    # Standardize this task's data
    scaler = StandardScaler()
    X_train_task_scaled = scaler.fit_transform(X_train_task).astype(np.float32)
    X_val_task_scaled = scaler.transform(X_val_task).astype(np.float32)

    y_train_task_np = y_train_task.values.astype(np.float32)
    y_val_task_np = y_val_task.values.astype(np.float32)

    # Train the GP on this task with the current initial hyperparameters
    results = create_and_train_gp(
        X_train_task_scaled,
        y_train_task_np,
        X_val_task_scaled,
        initial_lengthscale,
        initial_variance,
    )

    # "Meta-learn" (update) the initial hyperparameters based on this task's performance
    # In a real meta-learning scenario, you'd use a more sophisticated optimization algorithm
    # Here, we're doing a simple update:
    initial_lengthscale = (
        initial_lengthscale + meta_learning_rate * results["lengthscale"].mean().item()
    )
    initial_variance = (
        initial_variance + meta_learning_rate * results["variance"].mean().item()
    )

    print(f"Meta-train task {i+1}: R2 = {results['r2']:.3f}, RMSE = {results['rmse']:.3f}")

# Meta-testing
# Evaluate the GP on the meta-test set with the learned initial hyperparameters
scaler = StandardScaler()
X_meta_test_scaled = scaler.fit_transform(X_meta_test).astype(np.float32)
y_meta_test_np = y_meta_test.values.astype(np.float32)

meta_test_results = create_and_train_gp(
    X_meta_train,
    y_meta_train,
    X_meta_test_scaled,
    initial_lengthscale,
    initial_variance,
)  # Use ALL meta-train for final training

print(f"Meta-test: R2 = {meta_test_results['r2']:.3f}, RMSE = {meta_test_results['rmse']:.3f}")