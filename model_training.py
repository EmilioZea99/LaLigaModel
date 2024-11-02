import joblib
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Load the preprocessed data
X_test_scaled = joblib.load("X_test_scaled.pkl")
y_test = joblib.load("y_test.pkl")

# Load the best model found during hyperparameter tuning
best_model = joblib.load("best_model.pkl")

# Step 1: Make predictions on the test set
y_pred = best_model.predict(X_test_scaled)

# Step 2: Evaluate the model
# Accuracy Score
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.2f}")

# Classification Report
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Confusion Matrix
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))
