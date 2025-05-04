import pandas as pd
from sklearn.linear_model import LinearRegression
import pickle

# Sample weather data (you can use real datasets too)
data = {
    'humidity': [80, 60, 90, 50, 65],
    'pressure': [1012, 1010, 1005, 1020, 1015],
    'wind_speed': [10, 5, 7, 3, 8],
    'temperature': [30, 25, 28, 22, 26]
}

df = pd.DataFrame(data)

# Features and label
X = df[['humidity', 'pressure', 'wind_speed']]
y = df['temperature']

# Train the model
model = LinearRegression()
model.fit(X, y)

# Save the model
with open('model.pkl', 'wb') as f:
    pickle.dump(model, f)
