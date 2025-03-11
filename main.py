from sklearn.datasets import fetch_california_housing
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import RandomForestRegressor

import matplotlib.pyplot as plt
import pandas as pd

# Load dataset
data = fetch_california_housing()
X = data.data
y = data.target

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = RandomForestRegressor()
model.fit(X_train, y_train)

# Evaluasi
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
print(f"MSE: {mse:.2f}")

# Plot hubungan RM dengan harga rumah
plt.scatter(X_test[:, 5], y_test, color='blue', label='Aktual')  # RM adalah fitur ke-5
plt.scatter(X_test[:, 5], y_pred, color='red', label='Prediksi')  # Prediksi Random Forest
plt.xlabel('RM (Rata-rata Jumlah Kamar)')
plt.ylabel('Harga Rumah')
plt.title('Hubungan RM dengan Harga Rumah')
plt.legend()
plt.show()

# Plot prediksi vs aktual
plt.scatter(y_test, y_pred, color='blue')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=2)  # Garis diagonal
plt.xlabel('Aktual')
plt.ylabel('Prediksi')
plt.title('Prediksi vs Aktual (Random Forest)')
plt.show()


# Feature importance
feature_importance = pd.Series(model.feature_importances_, index=data.feature_names)
feature_importance.sort_values(ascending=False).plot(kind='barh')
plt.title('Feature Importance (Random Forest)')
plt.show()