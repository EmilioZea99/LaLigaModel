import joblib
import numpy as np

# Load the trained model and scaler
model = joblib.load("best_model.pkl")
scaler = joblib.load("scaler.pkl")

# Load the encoder for categorical features (teams)
encoder = joblib.load("encoder.pkl")

# Prepare input data for prediction
home_team = "Girona"
away_team = "Ath Bilbao"

# Normalize the input team names to match the training format
home_team = home_team.lower().strip()
away_team = away_team.lower().strip()

home_team_form = 4.0  # Example of form for last 5 matches (wins, draws, losses)
away_team_form = -1.0
home_team_goals_avg = 1.0
away_team_goals_avg = 1.54
home_team_conceded_avg = 1.27
away_team_conceded_avg = 1.0
home_team_advantage = 1  # Assuming it's the home team's advantage

# Encode categorical variables
try:
    home_team_encoded = encoder.transform([home_team])[0]
except ValueError:
    print(f"Warning: '{home_team}' not found in encoder classes.")
    home_team_encoded = -1  # Assign a default value for unseen team

try:
    away_team_encoded = encoder.transform([away_team])[0]
except ValueError:
    print(f"Warning: '{away_team}' not found in encoder classes.")
    away_team_encoded = -1  # Assign a default value for unseen team

# Create a feature vector
input_features = np.array(
    [
        [
            home_team_encoded,
            away_team_encoded,
            home_team_form,
            away_team_form,
            home_team_goals_avg,
            away_team_goals_avg,
            home_team_conceded_avg,
            away_team_conceded_avg,
            home_team_advantage,
        ]
    ]
)

# Scale the features using the fitted scaler
input_features_scaled = scaler.transform(input_features)

# Make a prediction
prediction = model.predict(input_features_scaled)

# Decode the prediction to get the match result
result_mapping = {0: "Home Win", 1: "Draw", 2: "Away Win"}
predicted_outcome = result_mapping[prediction[0]]

# Print the predicted outcome
print(f"Predicted outcome: {predicted_outcome}")
