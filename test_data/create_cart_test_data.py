"""
Create test data for CART analysis and Monte Carlo simulations.
"""

import numpy as np
import pandas as pd

np.random.seed(42)

# CART regression test data - simulates environmental model output
n = 500

# Input parameters (model parameters)
param1 = np.random.uniform(0.1, 2.0, n)      # Growth rate
param2 = np.random.uniform(10, 30, n)        # Temperature
param3 = np.random.uniform(0, 100, n)        # Light intensity
param4 = np.random.uniform(0.01, 0.5, n)     # Nutrient concentration
param5 = np.random.uniform(1, 10, n)         # Depth
param6 = np.random.uniform(0, 1, n)          # Substrate roughness
param7 = np.random.uniform(0.1, 5, n)        # Current velocity
param8 = np.random.uniform(0, 50, n)         # Turbidity

# Output - biomass prediction (non-linear with interactions)
# This simulates a complex ecological model
biomass = (
    50 * param1                               # Main effect of growth rate
    + 2 * param2                              # Temperature effect
    - 0.5 * param3                            # Light saturation
    + 100 * param4                            # Nutrient limitation
    - 5 * param5                              # Depth attenuation
    + 10 * param6 * param1                    # Interaction: substrate × growth
    - 3 * param7 * np.sqrt(param5)            # Interaction: current × depth
    + 0.1 * param2 * param4                   # Interaction: temp × nutrients
    + np.random.normal(0, 10, n)              # Random error
)

# Ensure positive biomass
biomass = np.maximum(biomass, 0)

cart_data = pd.DataFrame({
    'growth_rate': param1,
    'temperature': param2,
    'light_intensity': param3,
    'nutrient_conc': param4,
    'depth': param5,
    'substrate_roughness': param6,
    'current_velocity': param7,
    'turbidity': param8,
    'biomass': biomass
})

cart_data.to_csv('cart_biomass_model.csv', index=False)
print(f"Created cart_biomass_model.csv with {len(cart_data)} rows")
print(f"Columns: {list(cart_data.columns)}")
print(f"\nBiomass statistics:")
print(cart_data['biomass'].describe())
