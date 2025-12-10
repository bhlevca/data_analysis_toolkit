"""
Create comprehensive ML test datasets for regression and classification.
Generates larger, more complex datasets with clear patterns for demonstration.
"""
import numpy as np
import pandas as pd
from pathlib import Path

np.random.seed(42)

# Output directory
OUT_DIR = Path(__file__).parent


def create_regression_data():
    """
    Create a regression dataset: Predicting house prices based on multiple features.
    Non-linear relationships with some noise for realistic behavior.
    """
    n_train = 200
    n_predict = 50
    
    # Training data
    size_sqft = np.random.uniform(800, 4000, n_train)
    bedrooms = np.random.randint(1, 6, n_train)
    bathrooms = np.random.randint(1, 4, n_train)
    age_years = np.random.uniform(0, 50, n_train)
    distance_city = np.random.uniform(1, 30, n_train)
    lot_size = np.random.uniform(2000, 20000, n_train)
    garage_spaces = np.random.randint(0, 4, n_train)
    
    # Price formula with realistic relationships
    # Base: $50 per sqft, +$15k per bedroom, +$20k per bathroom
    # -$1k per year of age, -$2k per mile from city, +$5 per sqft lot, +$10k per garage
    price = (
        50 * size_sqft +
        15000 * bedrooms +
        20000 * bathrooms -
        1000 * age_years -
        2000 * distance_city +
        5 * lot_size +
        10000 * garage_spaces +
        50000  # base price
    )
    # Add noise (5% of price)
    price += np.random.normal(0, price * 0.05)
    price = np.maximum(price, 50000)  # minimum price
    
    train_df = pd.DataFrame({
        'size_sqft': np.round(size_sqft, 1),
        'bedrooms': bedrooms,
        'bathrooms': bathrooms,
        'age_years': np.round(age_years, 1),
        'distance_city_miles': np.round(distance_city, 1),
        'lot_size_sqft': np.round(lot_size, 0).astype(int),
        'garage_spaces': garage_spaces,
        'price': np.round(price, 0).astype(int)
    })
    
    # Prediction data (new houses to predict)
    size_sqft_p = np.random.uniform(800, 4000, n_predict)
    bedrooms_p = np.random.randint(1, 6, n_predict)
    bathrooms_p = np.random.randint(1, 4, n_predict)
    age_years_p = np.random.uniform(0, 50, n_predict)
    distance_city_p = np.random.uniform(1, 30, n_predict)
    lot_size_p = np.random.uniform(2000, 20000, n_predict)
    garage_spaces_p = np.random.randint(0, 4, n_predict)
    
    # Actual prices for prediction data (for evaluation)
    price_p = (
        50 * size_sqft_p +
        15000 * bedrooms_p +
        20000 * bathrooms_p -
        1000 * age_years_p -
        2000 * distance_city_p +
        5 * lot_size_p +
        10000 * garage_spaces_p +
        50000
    )
    price_p += np.random.normal(0, price_p * 0.05)
    price_p = np.maximum(price_p, 50000)
    
    predict_df = pd.DataFrame({
        'size_sqft': np.round(size_sqft_p, 1),
        'bedrooms': bedrooms_p,
        'bathrooms': bathrooms_p,
        'age_years': np.round(age_years_p, 1),
        'distance_city_miles': np.round(distance_city_p, 1),
        'lot_size_sqft': np.round(lot_size_p, 0).astype(int),
        'garage_spaces': garage_spaces_p,
        'price': np.round(price_p, 0).astype(int)  # Include actual for comparison
    })
    
    return train_df, predict_df


def create_classification_data():
    """
    Create a classification dataset: Customer segmentation into 4 categories.
    Features: age, income, spending_score, membership_years, purchase_frequency
    Classes: Budget, Standard, Premium, VIP
    """
    n_train = 300
    n_predict = 75
    
    def generate_customer_data(n):
        # Generate base features
        age = np.random.randint(18, 75, n)
        income = np.random.uniform(20000, 200000, n)
        spending_score = np.random.uniform(0, 100, n)
        membership_years = np.random.uniform(0, 15, n)
        purchase_frequency = np.random.randint(1, 52, n)  # purchases per year
        avg_transaction = np.random.uniform(20, 500, n)
        
        # Determine class based on features (with some overlap for realism)
        # VIP: high income, high spending, long membership
        # Premium: good income, decent spending
        # Standard: moderate everything
        # Budget: lower income, low spending
        
        customer_score = (
            0.3 * (income / 200000) +
            0.25 * (spending_score / 100) +
            0.2 * (membership_years / 15) +
            0.15 * (purchase_frequency / 52) +
            0.1 * (avg_transaction / 500)
        )
        
        # Add some noise to make it realistic
        customer_score += np.random.normal(0, 0.08, n)
        
        # Classify
        segment = np.where(customer_score >= 0.7, 'VIP',
                  np.where(customer_score >= 0.5, 'Premium',
                  np.where(customer_score >= 0.3, 'Standard', 'Budget')))
        
        # Also create numeric class for models that need it
        segment_code = np.where(segment == 'VIP', 3,
                       np.where(segment == 'Premium', 2,
                       np.where(segment == 'Standard', 1, 0)))
        
        return pd.DataFrame({
            'age': age,
            'annual_income': np.round(income, 0).astype(int),
            'spending_score': np.round(spending_score, 1),
            'membership_years': np.round(membership_years, 1),
            'purchase_frequency': purchase_frequency,
            'avg_transaction': np.round(avg_transaction, 2),
            'segment': segment,
            'segment_code': segment_code
        })
    
    train_df = generate_customer_data(n_train)
    predict_df = generate_customer_data(n_predict)
    
    return train_df, predict_df


def main():
    print("Creating ML test datasets...")
    
    # Regression
    reg_train, reg_predict = create_regression_data()
    reg_train.to_csv(OUT_DIR / 'ml_regression_train.csv', index=False)
    reg_predict.to_csv(OUT_DIR / 'ml_regression_predict.csv', index=False)
    print(f"✅ Regression training: {len(reg_train)} samples, {len(reg_train.columns)} columns")
    print(f"   Features: {list(reg_train.columns[:-1])}")
    print(f"   Target: price")
    print(f"✅ Regression prediction: {len(reg_predict)} samples")
    
    # Classification
    cls_train, cls_predict = create_classification_data()
    cls_train.to_csv(OUT_DIR / 'ml_classification_train.csv', index=False)
    cls_predict.to_csv(OUT_DIR / 'ml_classification_predict.csv', index=False)
    print(f"✅ Classification training: {len(cls_train)} samples, {len(cls_train.columns)} columns")
    print(f"   Features: {list(cls_train.columns[:-2])}")
    print(f"   Target: segment (4 classes: Budget, Standard, Premium, VIP)")
    print(f"✅ Classification prediction: {len(cls_predict)} samples")
    
    # Show class distribution
    print(f"\n📊 Class distribution in training:")
    print(cls_train['segment'].value_counts().sort_index())
    
    print("\n✅ All datasets created!")


if __name__ == '__main__':
    main()
