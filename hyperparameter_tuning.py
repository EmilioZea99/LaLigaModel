import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV

# Load the preprocessed data
X_train_scaled = joblib.load("X_train_scaled.pkl")
y_train = joblib.load("y_train.pkl")

# Initialize the Random Forest Classifier
model = RandomForestClassifier(random_state=42)

# Set up the hyperparameter grid
param_grid = {
    "n_estimators": [50, 100, 200],
    "max_depth": [5, 10, 15],
    "min_samples_split": [2, 5, 10],
    "min_samples_leaf": [1, 2, 4],
    "class_weight": ["balanced"],
}

# Initialize GridSearchCV
grid_search = GridSearchCV(
    model, param_grid, cv=5, n_jobs=-1, verbose=2, scoring="accuracy"
)

# Fit the model to find the best hyperparameters
grid_search.fit(X_train_scaled, y_train)

# Print the best parameters and best score
print(f"Best Parameters: {grid_search.best_params_}")
print(f"Best Cross-Validation Score: {grid_search.best_score_:.2f}")

# Save the best model
best_model = grid_search.best_estimator_
joblib.dump(best_model, "best_model.pkl")
