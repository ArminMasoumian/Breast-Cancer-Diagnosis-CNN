import numpy as np
from sklearn.model_selection import train_test_split

# Load the dataset
X = np.load('breast_cancer_images.npy')
y = np.load('breast_cancer_labels.npy')

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Preprocess the data
X_train = X_train / 255.0
X_test = X_test / 255.0
