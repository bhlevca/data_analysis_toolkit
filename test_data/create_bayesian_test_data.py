"""
Generate Bayesian test data with proper homo- and heteroscedastic responses.

True model:  y = 5.0 + 2.0*x1 - 1.5*x2 + 0.8*x3 + noise

Homoscedastic:   noise ~ N(0, sigma=2)                     (constant variance)
Heteroscedastic: noise ~ N(0, sigma=0.5*predictor_1)       (variance grows with x1)

predictor_1 is drawn from Uniform(0.5, 20) to give a wide dynamic range,
ensuring the heteroscedastic noise varies dramatically (std from ~0.25 to ~10).
"""
import numpy as np
import pandas as pd

np.random.seed(42)

n = 300

# Predictors — predictor_1 has wide range for clear heteroscedasticity
predictor_1 = np.random.uniform(0.5, 20, n)  # wide range for dramatic noise contrast
predictor_2 = np.random.normal(5, 2, n)      # mean=5, sd=2
predictor_3 = np.random.normal(0, 1, n)      # mean=0, sd=1

# True linear relationship
y_true = 5.0 + 2.0 * predictor_1 - 1.5 * predictor_2 + 0.8 * predictor_3

# Homoscedastic: constant noise sd = 2
noise_homo = np.random.normal(0, 2.0, n)
response_homo = y_true + noise_homo

# Heteroscedastic: noise sd proportional to predictor_1
# When predictor_1 ≈ 1 → sd ≈ 0.5,  when predictor_1 ≈ 20 → sd ≈ 10
noise_std_hetero = 0.5 * predictor_1
noise_hetero = np.random.normal(0, noise_std_hetero)
response_hetero = y_true + noise_hetero

df = pd.DataFrame({
    'predictor_1': predictor_1,
    'predictor_2': predictor_2,
    'predictor_3': predictor_3,
    'response_homoscedastic': response_homo,
    'response_heteroscedastic': response_hetero,
})

df.to_csv('test_data/bayesian_uncertainty_data.csv', index=False)
print(f"Saved {len(df)} rows to test_data/bayesian_uncertainty_data.csv")
print(f"\nDescriptive stats:")
print(df.describe().round(3))

# Verify heteroscedasticity
from sklearn.linear_model import LinearRegression
X = df[['predictor_1','predictor_2','predictor_3']].values
lr_homo = LinearRegression().fit(X, response_homo)
lr_hetero = LinearRegression().fit(X, response_hetero)
resid_homo = response_homo - lr_homo.predict(X)
resid_hetero = response_hetero - lr_hetero.predict(X)

bins = np.percentile(predictor_1, [0, 25, 50, 75, 100])
print(f"\nResidual std by predictor_1 quartile:")
for i in range(4):
    lo, hi = bins[i], bins[i+1] if i < 3 else np.inf
    mask = (predictor_1 >= lo) & (predictor_1 < hi)
    print(f"  Q{i+1} ({lo:.1f}-{hi:.1f}): homo={resid_homo[mask].std():.3f}  hetero={resid_hetero[mask].std():.3f}")
