import joblib
import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

# Load the dataset
csv_file_path = "la-liga-results-19952020/LaLiga_Matches.csv"
df = pd.read_csv(csv_file_path)

# 1. Convert 'FTR' to numeric values for rolling calculations
# 'H' (Home win) -> 1, 'A' (Away win) -> -1, 'D' (Draw) -> 0
ftr_mapping = {"H": 1, "A": -1, "D": 0}
df["FTR_numeric"] = df["FTR"].map(ftr_mapping)

# 2. Normalize team names to lowercase and strip spaces to ensure consistency
df["HomeTeam"] = df["HomeTeam"].str.lower().str.strip()
df["AwayTeam"] = df["AwayTeam"].str.lower().str.strip()

# 3. Calculate Recent Team Form
# Calculate the number of wins, losses, or draws for the last 5 matches for each team
# Group by 'HomeTeam' and compute the rolling form
home_form = (
    df.groupby("HomeTeam")["FTR_numeric"]
    .rolling(5, min_periods=1)
    .sum()
    .reset_index(level=0, drop=True)
)
df["HomeTeamForm"] = home_form

# Group by 'AwayTeam' and compute the rolling form
away_form = (
    df.groupby("AwayTeam")["FTR_numeric"]
    .rolling(5, min_periods=1)
    .sum()
    .reset_index(level=0, drop=True)
)
df["AwayTeamForm"] = away_form

# Fill NaNs in rolling calculations with 0s (for the start of the season)
df.fillna(0, inplace=True)

# 4. Calculate Average Goals Scored in Recent Matches
# Group by 'HomeTeam' and compute average goals scored in the last 5 matches
home_goals_avg = (
    df.groupby("HomeTeam")["FTHG"]
    .rolling(5, min_periods=1)
    .mean()
    .reset_index(level=0, drop=True)
)
df["HomeTeamGoalsAvg"] = home_goals_avg

# Group by 'AwayTeam' and compute average goals scored in the last 5 matches
away_goals_avg = (
    df.groupby("AwayTeam")["FTAG"]
    .rolling(5, min_periods=1)
    .mean()
    .reset_index(level=0, drop=True)
)
df["AwayTeamGoalsAvg"] = away_goals_avg

# 5. Calculate Average Goals Conceded in Recent Matches
# Group by 'HomeTeam' and compute average goals conceded in the last 5 matches
home_conceded_avg = (
    df.groupby("HomeTeam")["FTAG"]
    .rolling(5, min_periods=1)
    .mean()
    .reset_index(level=0, drop=True)
)
df["HomeTeamConcededAvg"] = home_conceded_avg

# Group by 'AwayTeam' and compute average goals conceded in the last 5 matches
away_conceded_avg = (
    df.groupby("AwayTeam")["FTHG"]
    .rolling(5, min_periods=1)
    .mean()
    .reset_index(level=0, drop=True)
)
df["AwayTeamConcededAvg"] = away_conceded_avg

# 6. Add Home/Away Advantage Feature
# Create a binary feature indicating if the match is at home (1) or away (0) for the stronger team
df["HomeTeamAdvantage"] = 1

# 7. Feature Engineering: Define features and target
features = df[
    [
        "HomeTeam",
        "AwayTeam",
        "HomeTeamForm",
        "AwayTeamForm",
        "HomeTeamGoalsAvg",
        "AwayTeamGoalsAvg",
        "HomeTeamConcededAvg",
        "AwayTeamConcededAvg",
        "HomeTeamAdvantage",
    ]
]
target = df["FTR"]

# 8. Encode categorical variables using the same encoder for both HomeTeam and AwayTeam
encoder = LabelEncoder()
df["HomeTeam"] = encoder.fit_transform(df["HomeTeam"])
df["AwayTeam"] = encoder.transform(df["AwayTeam"])

# Print the classes to verify that all teams are included
print("Teams in encoder:", encoder.classes_)

# Apply the same encoding to the features
features.loc[:, "HomeTeam"] = df["HomeTeam"]
features.loc[:, "AwayTeam"] = df["AwayTeam"]

# Encode the target variable
target = encoder.fit_transform(target)

# Save the encoder
joblib.dump(encoder, "encoder.pkl")

# 9. Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    features, target, test_size=0.2, random_state=42, stratify=target
)

# 10. Feature Scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 11. Apply SMOTE to balance classes in the training set
smote = SMOTE(random_state=42)
X_train_balanced, y_train_balanced = smote.fit_resample(X_train_scaled, y_train)

# Save the scaler
joblib.dump(scaler, "scaler.pkl")

# 12. Save the preprocessed data
joblib.dump(X_train_balanced, "X_train_scaled.pkl")
joblib.dump(X_test_scaled, "X_test_scaled.pkl")
joblib.dump(y_train_balanced, "y_train.pkl")
joblib.dump(y_test, "y_test.pkl")
