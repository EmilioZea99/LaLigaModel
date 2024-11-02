import pandas as pd

# Load the dataset
df = pd.read_csv(
    "la-liga-results-19952020/LaLiga_Matches.csv"
)  # Adjust path and filename if needed

# Display general info about the dataset
print(df.info())
print(df.head())

# Display statistical summary of the dataset
print(df.describe())

# Check for missing values
print(df.isnull().sum())

# Display all unique teams in the dataset
home_teams = df["HomeTeam"].unique()
away_teams = df["AwayTeam"].unique()

# Combine both home and away teams to get the full list of unique teams
all_teams = set(home_teams).union(set(away_teams))

print("\nList of all unique teams in the dataset:")
for team in sorted(all_teams):
    print(team)
