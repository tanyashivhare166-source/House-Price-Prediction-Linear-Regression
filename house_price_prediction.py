# Import Libraries
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score

# Load Dataset
data = pd.read_csv("Housing.csv")

# Show First 5 Rows
print("First 5 Rows:")
print(data.head())

# Select Features (Input)
X = data[['area', 'bedrooms', 'bathrooms']]

# Select Target (Output)
y = data['price']

# Split Dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Create Model
model = LinearRegression()

# Train Model
model.fit(X_train, y_train)

# Predict
y_pred = model.predict(X_test)

# Accuracy
print("\nR2 Score:", r2_score(y_test, y_pred))
print("Mean Absolute Error:", mean_absolute_error(y_test, y_pred))

# Predict New House Price
area = 5000
bedrooms = 3
bathrooms = 2

prediction = model.predict([[area, bedrooms, bathrooms]])

print("\nPredicted Price =", prediction[0])

# Graph
plt.scatter(y_test, y_pred)
plt.xlabel("Actual Price")
plt.ylabel("Predicted Price")
plt.title("Actual vs Predicted House Price")
plt.show()