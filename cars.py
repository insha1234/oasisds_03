import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import numpy as np

# Generating a synthetic dataset
np.random.seed(42)
data = pd.DataFrame({
    'Year': np.random.randint(2000, 2023, 100),
    'Mileage': np.random.randint(5000, 100000, 100),
    'Horsepower': np.random.randint(100, 500, 100),
    'MPG_City': np.random.uniform(10, 50, 100),
    'MPG_Highway': np.random.uniform(20, 60, 100),
    'MSRP': np.random.randint(20000, 80000, 100)
})

# Explore the dataset
print("Dataset Overview:")
print(data.head())

# Data preprocessing
selected_features = ['Year', 'Mileage', 'Horsepower', 'MPG_City', 'MPG_Highway']

# Split the data into features (X) and target variable (y)
X = data[selected_features]
y = data['MSRP']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a RandomForestRegressor
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = model.predict(X_test)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("\nModel Evaluation:")
print(f"Mean Squared Error: {mse:.2f}")
print(f"R-squared: {r2:.2f}")

# Visualize predicted vs. actual prices
plt.scatter(y_test, y_pred)
plt.xlabel("Actual Prices")
plt.ylabel("Predicted Prices")
plt.title("Actual Prices vs. Predicted Prices")
plt.show()
