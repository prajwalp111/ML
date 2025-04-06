import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression  # Corrected import statement

# Load the dataset
students_data = pd.read_csv("students_data.csv")

# Create the X (class size) and y (test score) arrays
X = students_data[['class_size']]
y = students_data['test_score']

# Fit the model to the data
reg = LinearRegression().fit(X, y)

# Print the coefficients
print("Intercept:", reg.intercept_)
print("Coefficient:", reg.coef_)  # Fixed print statement
